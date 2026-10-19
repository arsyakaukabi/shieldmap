"""Exact narrow-beam polyenergetic transmission and its inversion.

T(x) = Σ_k w_k exp(−μ_k x) / Σ_k w_k,  with w_k = Φ_k · R(E_k).

The required thickness solves ln T(x) = ln T* with Brent's method (a vectorised transcription of
SciPy's ``brentq``, scipy/optimize/Zeros/brentq.c), run simultaneously for many materials. μ is
never averaged. The bracket is exact: since μ_min ≤ μ_k ≤ μ_max on the spectrum support,
ln(1/T*)/μ_max ≤ x* ≤ ln(1/T*)/μ_min.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def log_transmission(mu: np.ndarray, w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """ln T for each row of mu (n × K, 1/cm) at thickness x (n,) cm; w (K,) weights ≥ 0."""
    lw = np.log(w) - np.log(w.sum())
    return logsumexp(lw[None, :] - mu * x[:, None], axis=1)


def brentq_vec(f, a: np.ndarray, b: np.ndarray, xtol: np.ndarray, rtol: float,
               maxiter: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Brent root finder. ``f(x, idx)`` evaluates the function for rows ``idx``.

    Requires f(a)·f(b) ≤ 0 elementwise. Returns (root, iterations).
    """
    n = a.size
    xpre = a.astype(float).copy(); xcur = b.astype(float).copy()
    allidx = np.arange(n)
    fpre = f(xpre, allidx); fcur = f(xcur, allidx)
    if np.any(fpre * fcur > 0):
        raise ValueError("root not bracketed")
    root = np.full(n, np.nan); iters = np.zeros(n, dtype=int)
    done = fpre == 0
    root[done] = xpre[done]
    d2 = (fcur == 0) & ~done
    root[d2] = xcur[d2]; done |= d2
    xblk = np.zeros(n); fblk = np.zeros(n); spre = np.zeros(n); scur = np.zeros(n)
    xtol = np.broadcast_to(np.asarray(xtol, dtype=float), (n,)).copy()
    act = ~done
    for it in range(maxiter):
        i = np.where(act)[0]
        if i.size == 0:
            break
        # 1. keep the root bracketed by [xcur, xblk]
        sc = fpre[i] * fcur[i] < 0
        j = i[sc]
        xblk[j] = xpre[j]; fblk[j] = fpre[j]; spre[j] = scur[j] = xcur[j] - xpre[j]
        # 2. make xcur the best estimate
        sw = np.abs(fblk[i]) < np.abs(fcur[i])
        j = i[sw]
        xpre[j] = xcur[j]; xcur[j] = xblk[j]; xblk[j] = xpre[j]
        fpre[j] = fcur[j]; fcur[j] = fblk[j]; fblk[j] = fpre[j]
        delta = (xtol[i] + rtol * np.abs(xcur[i])) / 2
        sbis = (xblk[i] - xcur[i]) / 2
        conv = (fcur[i] == 0) | (np.abs(sbis) < delta)
        jc = i[conv]
        root[jc] = xcur[jc]; iters[jc] = it; act[jc] = False
        keep = ~conv
        i, delta, sbis = i[keep], delta[keep], sbis[keep]
        if i.size == 0:
            break
        # 3. interpolation step if the previous step was large and f decreased
        interp = (np.abs(spre[i]) > delta) & (np.abs(fcur[i]) < np.abs(fpre[i]))
        stry = np.zeros(i.size)
        with np.errstate(divide="ignore", invalid="ignore"):
            sec = interp & (xpre[i] == xblk[i])
            ii = i[sec]
            stry[sec] = -fcur[ii] * (xcur[ii] - xpre[ii]) / (fcur[ii] - fpre[ii])
            iq = interp & ~sec
            ii = i[iq]
            dpre = (fpre[ii] - fcur[ii]) / (xpre[ii] - xcur[ii])
            dblk = (fblk[ii] - fcur[ii]) / (xblk[ii] - xcur[ii])
            stry[iq] = -fcur[ii] * (fblk[ii] * dblk - fpre[ii] * dpre) / (dblk * dpre * (fblk[ii] - fpre[ii]))
        good = interp & (2 * np.abs(stry) < np.minimum(np.abs(spre[i]), 3 * np.abs(sbis) - delta))
        newspre = np.where(good, scur[i], sbis)
        newscur = np.where(good, stry, sbis)
        spre[i] = newspre; scur[i] = newscur
        # 4. step
        xpre[i] = xcur[i]; fpre[i] = fcur[i]
        big = np.abs(scur[i]) > delta
        xcur[i] = np.where(big, xcur[i] + scur[i], xcur[i] + np.where(sbis > 0, delta, -delta))
        fcur[i] = f(xcur[i], i)
    if np.any(act):
        raise RuntimeError(f"Brent did not converge for {int(act.sum())} rows in {maxiter} iterations")
    return root, iters


def solve_thickness(mu: np.ndarray, w: np.ndarray, t_target: float | np.ndarray,
                    xtol_rel: float = 1e-10) -> np.ndarray:
    """Thickness x (cm) with T(x) = t_target for every row of mu (n × K, 1/cm)."""
    mu = np.atleast_2d(np.asarray(mu, dtype=float))
    w = np.asarray(w, dtype=float)
    if mu.shape[1] != w.size:
        raise ValueError("mu and w disagree on the number of bins")
    supp = w > 0
    mu, w = mu[:, supp], w[supp]
    n = mu.shape[0]
    lt = np.log(np.broadcast_to(np.asarray(t_target, dtype=float), (n,)))
    if np.any(lt >= 0):
        raise ValueError("target transmission must be < 1")
    mu_max = mu.max(axis=1); mu_min = mu.min(axis=1)
    if np.any(mu_min <= 0):
        raise ValueError("non-positive attenuation coefficient")
    a = -lt / mu_max; b = -lt / mu_min
    # A single line (or flat μ) gives a == b; widen the bracket so Brent still does the solving.
    flat = b <= a * (1 + 1e-12)
    a = np.where(flat, 0.5 * a, a); b = np.where(flat, 2.0 * b, b)

    def f(x, idx):
        return log_transmission(mu[idx], w, x) - lt[idx]

    x, _ = brentq_vec(f, a, b, xtol=1e-14 * b, rtol=xtol_rel)
    return x
