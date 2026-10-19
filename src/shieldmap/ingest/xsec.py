# Licence: the evaluation path below is a port of XCOM.f (NIST XCOM; Copyright 1999, Martin J. Berger: "Permission is given
# to use the program for non-commercial purposes"). This file is excluded from the MIT grant to that extent; see NOTICE.
"""NIST XCOM 3.1 element cross sections, evaluated with XCOM's own interpolation.

Source: the standalone XCOM 3.1 distribution (Berger & Hubbell, 23 June 1999), downloaded once
as ``XCOM.tar.gz`` (config/sources/nist_xcom.yaml). Each ``MDATX3.nnn`` file holds, for one
element, partial cross sections (b/atom) on the 80-point standard grid plus edge points, the edge
energies, and supplementary photoabsorption tables for every inter-edge segment below the K edge.

This module is a line-by-line port of the evaluation path in ``XCOM.f`` (main program, labels
220–600, and subroutines SCOF/BSPOL/BLIN), so that values between tabulated energies are those the
XCOM program itself would print:

* coherent, incoherent:     natural cubic spline in (ln E, ln σ) over the edge-deduplicated list
* pair (nuclear/electron):  natural cubic spline in ln E of ln[σ / ((E − E_th)/E)³], with XCOM's
                            three-point threshold extrapolation; σ = 0 at or below threshold
* photoabsorption, E ≥ K:   natural cubic spline in (ln E, ln σ) over the points at/above K
* photoabsorption, E < K:   linear interpolation in (ln E, ln σ) in the segment table that
                            contains E (an energy equal to an edge takes the above-edge segment)

Conversion b/atom → cm²/g uses AVOG = 0.60221367 (XCOM.f) and the *same* atomic masses that define
the compound mass fractions and the MP densities (pymatgen, IUPAC; rev.), so that
Σ wᵢ(μ/ρ)ᵢ = N_A Σ nᵢσᵢ / Σ nᵢAᵢ holds exactly per atom. XCOM's own ATWTS.DAT weights are kept for
reference (``atwt_xcom``); for stable elements the two sets differ by ≤ 0.05 %, but for elements without
a stable isotope they are different mass-number conventions (e.g. Pu 239.05 vs 244).
"""
from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from shieldmap import config, grid, paths
from shieldmap.ingest import fetch

SOURCE_ID = "nist_xcom"
URL = "https://physics.nist.gov/PhysRefData/Xcom/XCOM.tar.gz"
RAW = paths.RAW / "xcom"
OUT = paths.INTERIM / "elements"

AVOG = 0.60221367            # 1e24 mol^-1 (barn → cm² folded in), XCOM.f DATA statement
EPAIR1 = 1.022007e6          # eV, pair production threshold in the nuclear field (XCOM.f)
EPAIR2 = 2.044014e6          # eV, threshold in the electron field (triplet)
PARTIALS = ("coh", "incoh", "pe", "pair_nuc", "pair_el")


# ----------------------------------------------------------------------------------------------
# retrieval
# ----------------------------------------------------------------------------------------------
def fetch_package() -> tuple[Path, dict]:
    """Download (once) and extract the XCOM 3.1 package. Returns (extract dir, file record)."""
    rec = fetch.download(URL, RAW / "XCOM.tar.gz", source_id=SOURCE_ID)
    xdir = RAW / "extracted"
    if not (xdir / "XCOM" / "MDATX3.100").exists():
        with tarfile.open(RAW / "XCOM.tar.gz", "r:gz") as tf:
            members = [m for m in tf.getmembers()
                       if m.isfile() and "/.rsrc/" not in m.name and not m.name.startswith("/")
                       and ".." not in Path(m.name).parts]
            tf.extractall(xdir, members=members, filter="data")
    return xdir / "XCOM", rec


def read_atwts(xcom_dir: Path) -> np.ndarray:
    """Atomic weights from ATWTS.DAT (Fortran DATA statement); index 0 ↔ Z = 1."""
    text = (xcom_dir / "ATWTS.DAT").read_text()
    lines = text.splitlines()
    body = []
    for ln in lines:
        if "DATA" in ln:
            ln = ln.split("/", 1)[1] if "/" in ln else ""
        elif len(ln) > 6:
            ln = ln[6:]                    # drop the continuation column
        body.append(ln)
    nums = re.findall(r"\d+\.\d*", " ".join(body))
    w = np.array([float(x) for x in nums])
    if w.size != 100:
        raise ValueError(f"ATWTS.DAT: expected 100 weights, found {w.size}")
    return w


def read_enb(xcom_dir: Path) -> np.ndarray:
    """XCOM's 80-point standard energy grid (eV) from ENB.DAT."""
    text = (xcom_dir / "ENB.DAT").read_text()
    nums = re.findall(r"\d+\.\d*E[+-]\d+", text)
    e = np.array([float(x) for x in nums])
    if e.size != 80:
        raise ValueError(f"ENB.DAT: expected 80 energies, found {e.size}")
    return e


# ----------------------------------------------------------------------------------------------
# one element
# ----------------------------------------------------------------------------------------------
@dataclass
class XcomElement:
    Z: int
    atwt_file: float
    atwt_xcom: float                             # ATWTS.DAT value (reference only)
    mass: float                                  # atomic mass used for b/atom → cm²/g (pymatgen)
    e: np.ndarray                                # eV, full list incl. both sides of each edge
    coh: np.ndarray
    incoh: np.ndarray
    pe: np.ndarray
    pair_nuc: np.ndarray
    pair_el: np.ndarray
    edge_labels: list[str] = field(default_factory=list)   # descending energy (file order, K first)
    edge_energies: np.ndarray = field(default_factory=lambda: np.zeros(0))  # eV, descending
    edge_index: list[int] = field(default_factory=list)    # 0-based index of above-edge point
    seg_e: list[np.ndarray] = field(default_factory=list)  # eV; seg k covers [edge_asc[k-1], edge_asc[k])
    seg_pe: list[np.ndarray] = field(default_factory=list)

    # -- construction ------------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: Path, atwts: np.ndarray, masses: np.ndarray) -> "XcomElement":
        text = path.read_text()
        # MDATX3.042 ends with a DOS end-of-file byte (Ctrl-Z) that the Fortran reader never reaches
        text = text.rstrip().rstrip("\x1a")
        if "\x1a" in text:
            raise ValueError(f"{path.name}: unexpected Ctrl-Z inside the data")
        tok = text.split()
        pos = 0

        def take(n, conv=float):
            nonlocal pos
            out = [conv(t) for t in tok[pos:pos + n]]
            pos += n
            return out

        z = int(tok[0]); atwt_file = float(tok[1]); pos = 2
        maxedg, maxe = take(2, int)
        idg_file, labels, edgen_file = [], [], []
        if maxedg > 0:
            idg_file = take(maxedg, int)          # file order: K first (descending energy)
            labels = take(maxedg, str)            # file order: K, L1, ...
            edgen_file = take(maxedg)             # eV, descending
        e = np.array(take(maxe))
        coh, incoh, pe, pn, pel = (np.array(take(maxe)) for _ in range(5))
        seg_e, seg_pe = [], []
        if maxedg > 0:
            (lax,) = take(1, int)
            kmx = take(lax, int)
            eng = [np.array(take(k)) * 1.0e6 for k in kmx]      # MeV → eV
            phc = [np.array(take(k)) for k in kmx]
            seg_e, seg_pe = eng, phc
            if lax != maxedg:
                raise ValueError(f"Z={z}: LAX={lax} != MAXEDG={maxedg}")
        if pos != len(tok):
            raise ValueError(f"Z={z}: {len(tok) - pos} unparsed tokens in {path.name}")
        if not np.all(np.diff(e) >= 0):
            raise ValueError(f"Z={z}: energies not ascending")
        el = cls(Z=z, atwt_file=atwt_file, atwt_xcom=float(atwts[z - 1]), mass=float(masses[z - 1]),
                 e=e, coh=coh, incoh=incoh,
                 pe=pe, pair_nuc=pn, pair_el=pel, edge_labels=labels,
                 edge_energies=np.array(edgen_file, dtype=float),
                 edge_index=[i - 1 for i in idg_file], seg_e=seg_e, seg_pe=seg_pe)
        el._prepare()
        return el

    # -- XCOM preprocessing (XCOM.f labels 310–437) -------------------------------------------
    def _prepare(self) -> None:
        below = {i - 1 for i in self.edge_index}           # below-edge duplicates are dropped
        keep = np.array([m not in below for m in range(self.e.size)])
        e = self.e[keep].copy()
        for name in ("coh", "incoh", "pe", "pair_nuc", "pair_el"):
            setattr(self, f"_{name}", getattr(self, name)[keep].copy())
        # XCOM overwrites the two threshold nodes of the standard grid with exact thresholds
        # (E(51) = EPAIR2, E(55) = EPAIR1 after reversal); this also shifts the spline knots.
        i1 = int(np.argmin(np.abs(e - 1.022e6))); i2 = int(np.argmin(np.abs(e - 2.044e6)))
        if not (np.isclose(e[i1], 1.022e6) and np.isclose(e[i2], 2.044e6)):
            raise ValueError(f"Z={self.Z}: threshold nodes not found")
        if e.size - i1 != 55 or e.size - i2 != 51:
            raise ValueError(f"Z={self.Z}: unexpected position of threshold nodes")
        e[i1] = EPAIR1; e[i2] = EPAIR2
        self._e = e
        le = np.log(e)
        self._sp_coh = CubicSpline(le, np.log(self._coh), bc_type="natural")
        self._sp_incoh = CubicSpline(le, np.log(self._incoh), bc_type="natural")

        # pair production: transformed quantity on the points above threshold + extrapolated node
        def pair_spline(sig, i_th, e_th, n_above):
            idx = np.arange(i_th + 1, e.size)
            if idx.size != n_above:
                raise ValueError(f"Z={self.Z}: {idx.size} points above {e_th} eV, expected {n_above}")
            term = (e[idx] - e_th) / e[idx]
            pr = np.log(sig[idx] / term ** 3)                     # ascending energy
            # XCOM (descending order): PR(N+1) = 3.006275 PR(N) - 2.577757 PR(N-1) + 0.571482 PR(N-2)
            # where PR(N) is the lowest-energy point above threshold.
            pr_th = 3.006275 * pr[0] - 2.577757 * pr[1] + 0.571482 * pr[2]
            return CubicSpline(np.concatenate([[np.log(e_th)], le[idx]]),
                               np.concatenate([[pr_th], pr]), bc_type="natural")

        self._sp_pn = pair_spline(self._pair_nuc, i1, EPAIR1, 54)
        self._sp_pe_el = pair_spline(self._pair_el, i2, EPAIR2, 50)

        if self.edge_labels:
            k_above = self.edge_index[0]                         # file order: K first
            # number of points at/above the K edge in the deduplicated list (MAXK in XCOM.f)
            n_k = int(np.sum(keep[k_above:]))
            self._sp_pe_hi = CubicSpline(le[-n_k:], np.log(self._pe[-n_k:]), bc_type="natural")
            self._ecut = float(self.edge_energies[0])            # K-edge energy (eV)
            self._edges_asc = self.edge_energies[::-1].copy()    # EDGEN(1..MAXEDG)
            self._seg_le = [np.log(s) for s in self.seg_e]
            self._seg_lp = [np.log(p) for p in self.seg_pe]
        else:
            self._sp_pe_hi = CubicSpline(le, np.log(self._pe), bc_type="natural")
            self._ecut = 0.0

    # -- evaluation ---------------------------------------------------------------------------
    def cross_sections_b(self, e_ev: np.ndarray) -> dict[str, np.ndarray]:
        """Partial cross sections (b/atom) at energies e_ev (eV)."""
        e_ev = np.atleast_1d(np.asarray(e_ev, dtype=float))
        le = np.log(e_ev)
        out = {"coh": np.exp(self._sp_coh(le)), "incoh": np.exp(self._sp_incoh(le))}
        pn = np.zeros_like(e_ev); pel = np.zeros_like(e_ev)
        m1 = e_ev > EPAIR1
        pn[m1] = ((e_ev[m1] - EPAIR1) / e_ev[m1]) ** 3 * np.exp(self._sp_pn(le[m1]))
        m2 = e_ev > EPAIR2
        pel[m2] = ((e_ev[m2] - EPAIR2) / e_ev[m2]) ** 3 * np.exp(self._sp_pe_el(le[m2]))
        out["pair_nuc"], out["pair_el"] = pn, pel
        pe = np.empty_like(e_ev)
        hi = e_ev >= self._ecut
        pe[hi] = np.exp(self._sp_pe_hi(le[hi]))
        if (~hi).any():
            # IMP = smallest i with T < EDGEN(i)  (an energy equal to an edge goes above it)
            imp = np.searchsorted(self._edges_asc, e_ev[~hi], side="right")
            vals = np.empty(imp.size)
            for k in np.unique(imp):
                sel = imp == k
                vals[sel] = np.exp(_blin(le[~hi][sel], self._seg_le[k], self._seg_lp[k]))
            pe[~hi] = vals
        out["pe"] = pe
        return out

    def mu_rho(self, e_ev: np.ndarray) -> dict[str, np.ndarray]:
        """Partial and total mass attenuation coefficients (cm²/g)."""
        cs = self.cross_sections_b(e_ev)
        f = AVOG / self.mass
        out = {k: v * f for k, v in cs.items()}
        out["total_wocoh"] = out["incoh"] + out["pe"] + out["pair_nuc"] + out["pair_el"]
        out["total_wcoh"] = out["total_wocoh"] + out["coh"]
        return out

    def edges_keV(self) -> list[tuple[str, float]]:
        return [(lab, en / 1000.0) for lab, en in zip(self.edge_labels, self.edge_energies)]


def _blin(s: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """XCOM BLIN: piecewise-linear interpolation with end-interval extrapolation (x ascending)."""
    if x.size < 2:
        raise ValueError("segment with fewer than 2 points")
    j = np.clip(np.searchsorted(x, s, side="right") - 1, 0, x.size - 2)
    return y[j] + (s - x[j]) * (y[j + 1] - y[j]) / (x[j + 1] - x[j])


# ----------------------------------------------------------------------------------------------
# all elements
# ----------------------------------------------------------------------------------------------
def atomic_masses() -> np.ndarray:
    """pymatgen (IUPAC) atomic masses for Z = 1–100, the masses behind Table B mass fractions."""
    import warnings
    from pymatgen.core import Element
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.array([float(Element.from_Z(z).atomic_mass) for z in range(1, 101)])


def load_elements(xcom_dir: Path | None = None) -> dict[int, XcomElement]:
    if xcom_dir is None:
        xcom_dir, _ = fetch_package()
    atwts = read_atwts(xcom_dir)
    masses = atomic_masses()
    els = {}
    for z in range(1, config.get("elements.z_max") + 1):
        el = XcomElement.from_file(xcom_dir / f"MDATX3.{z:03d}", atwts, masses)
        if el.Z != z:
            raise ValueError(f"MDATX3.{z:03d} holds Z={el.Z}")
        els[z] = el
    return els


def write_source(rec: dict, xcom_dir: Path) -> None:
    enb = read_enb(xcom_dir)
    prior = (fetch.read_source(SOURCE_ID) or {}).get("files", [{}])[0]
    if prior.get("retrieved_utc_from") and prior.get("retrieved_utc") == rec.get("retrieved_utc"):
        rec = {**rec, "retrieved_utc_from": prior["retrieved_utc_from"]}
    if "retrieved_utc" not in rec:
        # first retrieval happened before this record existed: use the file's write time, marked
        import datetime as _dt
        t = (paths.ROOT / rec["path"]).stat().st_mtime
        rec = {**rec, "retrieved_utc": _dt.datetime.fromtimestamp(t, _dt.timezone.utc).replace(microsecond=0).isoformat(),
               "retrieved_utc_from": "file mtime (no prior record)"}
    fetch.write_source(SOURCE_ID, {
        "title": "XCOM: Photon Cross Sections Database — standalone program and data, version 3.1",
        "citation": ("M. J. Berger, J. H. Hubbell, S. M. Seltzer, J. Chang, J. S. Coursey, R. Sukumar, "
                     "D. S. Zucker, and K. Olsen, XCOM: Photon Cross Section Database (version 1.5), "
                     "NIST Standard Reference Database 8 (XGAM), doi:10.18434/T48G6X, National Institute "
                     "of Standards and Technology, Gaithersburg, MD (2010); downloadable version 3.1 "
                     "(M. J. Berger and J. H. Hubbell, 23 June 1999). Originally published as M. J. Berger "
                     "and J. H. Hubbell, XCOM: Photon Cross Sections on a Personal Computer, NBSIR 87-3597 "
                     "(1987)."),
        "data_class": "P1",
        "retrieval_method": "bulk download of the standalone distribution (not the web form)",
        "url": URL,
        "landing_page": "https://physics.nist.gov/PhysRefData/Xcom/Text/download.html",
        "files": [rec],
        "licence": ("(c) 1990, 1998 U.S. Secretary of Commerce; XCOM.f grants use for non-commercial "
                    "purposes; NIST SRD records: https://www.nist.gov/open/license"),
        "notes": ("Mac resource-fork entries (.rsrc/) in the tarball are ignored. Interpolation is a "
                  "port of XCOM.f (see src/shieldmap/ingest/xsec.py)."),
        "data": {"standard_grid_keV": [float(x) / 1000.0 for x in enb]},
    })


def build() -> None:
    xcom_dir, rec = fetch_package()
    write_source(rec, xcom_dir)
    els = load_elements(xcom_dir)
    e_keV = grid.energy_grid_keV()
    n_e = e_keV.size
    arrays = {k: np.zeros((100, n_e)) for k in (*PARTIALS, "total_wcoh", "total_wocoh")}
    edge_rows = []
    for z, el in els.items():
        mr = el.mu_rho(e_keV * 1000.0)
        for k in arrays:
            arrays[k][z - 1] = mr[k]
        for lab, en in el.edges_keV():
            edge_rows.append({"Z": z, "shell": lab, "energy_keV": en})
    paths.ensure(OUT)
    np.savez(OUT / "xsec_grid.npz", energy_keV=e_keV, Z=np.arange(1, 101),
             atwt_xcom=np.array([els[z].atwt_xcom for z in range(1, 101)]),
             mass_used=np.array([els[z].mass for z in range(1, 101)]), **arrays)
    pd.DataFrame(edge_rows).to_parquet(OUT / "edges.parquet", index=False)


if __name__ == "__main__":
    build()
