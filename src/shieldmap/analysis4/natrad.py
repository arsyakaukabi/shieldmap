"""Natural-radioactivity flags.

  strict    an element with no stable isotope (Table B ``contains_radioactive``; stages 1-3)       PRIMARY
  moderate  strict, plus every element whose natural specific activity exceeds that of natural potassium
  named     strict + {Lu, Re, La}

Natural specific activity of element X (Bq per g of X):
    A_X = Σ_i x_i · (N_A / M_X) · ln 2 / T½,i
with x_i the IUPAC isotopic abundances shipped by ``mendeleev`` (the same package and route as the crustal
abundance table, config/sources/crustal_abundance.yaml), M_X the mendeleev standard atomic weight, and T½,i
the ENSDF adopted ground-state half-life (NNDC distribution data/raw/ensdf/ensdf_261001.zip, already
checksummed in config/sources/ensdf_cs137.yaml). A lower-limit half-life gives an upper bound on A.
mendeleev's own half-life column is not used.
"""
from __future__ import annotations

import math
import re
import zipfile
from decimal import Decimal
from functools import lru_cache

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis3 import common as c3
from shieldmap.analysis4 import common
from shieldmap.ingest import references as ens

ENSDF_ZIP = paths.RAW / "ensdf" / "ensdf_261001.zip"
N_A = 6.02214076e23
YEAR_S = 365.2422 * 86400.0
NAMED = ("Lu", "Re", "La")
REPORT = ("K", "La", "In", "Te", "Gd", "Hf", "Os", "Pt", "Rb", "Sm", "Lu", "Re", "Nd", "Ca", "V", "Bi", "W", "Cd",
          "Se", "Zr", "Mo", "Ba", "Ge", "Xe", "Kr", "Te", "Ta", "Eu")
_UNIT_S = {"Y": YEAR_S, "D": 86400.0, "H": 3600.0, "M": 60.0, "S": 1.0, "MS": 1e-3, "US": 1e-6, "NS": 1e-9,
           "PS": 1e-12, "FS": 1e-15, "AS": 1e-18}
_T = re.compile(r"^\s*([<>]=?|GE|LE|GT|LT|AP|~)?\s*([0-9.]+(?:E[+-]?\d+)?)\s*([A-Z]+)\s*$")


def _member(z: zipfile.ZipFile, a: int) -> str:
    name = next(n for n in z.namelist() if n.lower().endswith(f"ensdf.{a:03d}"))
    return z.read(name).decode("latin-1")


def _nucid(a: int, sym: str) -> str:
    return f"{a:>3d}{sym.upper():<2s}"


@lru_cache(maxsize=1)
def _zip_text() -> dict:
    return {}


def ensdf_half_life(a: int, sym: str) -> dict:
    """Adopted ground-state half-life of (A, element) from ENSDF: {'t_half_s', 'stable', 'limit', 'raw'}."""
    cache = _zip_text()
    if a not in cache:
        with zipfile.ZipFile(ENSDF_ZIP) as z:
            cache[a] = _member(z, a)
    nucid = _nucid(a, sym)
    for ds in ens.split_datasets(cache[a]):
        head = ds[0].ljust(80)
        if head[:5].upper() != nucid or not ens._col(head, 10, 39).upper().startswith("ADOPTED LEVELS"):
            continue
        for line in ds:
            cont, com, rt = ens._record_type(line)
            if rt != "L" or cont != " " or com != " " or line.ljust(80)[:5].upper() != nucid:
                continue
            e = ens._col(line, 10, 19)
            if ens.ensdf_value(e) is None or Decimal(e) != 0:
                continue
            raw = ens._col(line, 40, 49)
            dt = ens._col(line, 50, 55)
            if raw.upper().startswith("STABLE"):
                return {"t_half_s": math.inf, "stable": True, "limit": "", "raw": raw}
            m = _T.match(raw.upper())
            if not m or m.group(3) not in _UNIT_S:
                return {"t_half_s": math.nan, "stable": False, "limit": "unparsed", "raw": raw}
            lim = (m.group(1) or "").strip()
            if dt.upper() in ("GT", "GE", "LT", "LE", "AP"):
                lim = lim or dt.upper()
            return {"t_half_s": float(m.group(2)) * _UNIT_S[m.group(3)], "stable": False, "limit": lim, "raw": raw}
        return {"t_half_s": math.nan, "stable": False, "limit": "no ground-state L record", "raw": ""}
    return {"t_half_s": math.nan, "stable": False, "limit": "no adopted-levels dataset", "raw": ""}


def isotope_table(symbols) -> pd.DataFrame:
    from mendeleev import element
    recs = []
    for s in symbols:
        el = element(s)
        for iso in el.isotopes:
            if not iso.abundance:
                continue
            hl = ensdf_half_life(int(iso.mass_number), s)
            recs.append({"element": s, "Z": el.atomic_number, "A": int(iso.mass_number),
                         "abundance_pct": float(iso.abundance), "atomic_weight": float(el.atomic_weight),
                         "ensdf_t_half_raw": hl["raw"], "t_half_s": hl["t_half_s"], "stable": hl["stable"],
                         "t_half_limit": hl["limit"]})
    t = pd.DataFrame(recs)
    lam = np.where(np.isfinite(t.t_half_s) & (t.t_half_s > 0), math.log(2) / t.t_half_s, 0.0)
    t["specific_activity_Bq_per_g_element"] = t.abundance_pct / 100.0 * N_A / t.atomic_weight * lam
    return t


def element_activity(symbols) -> pd.DataFrame:
    iso = isotope_table(symbols)
    recs = []
    limits = (">", ">=", "GT", "GE")
    for s, d in iso.groupby("element"):
        recs.append({"element": s, "Z": int(d.Z.iloc[0]),
                     "A_Bq_per_g": float(d.specific_activity_Bq_per_g_element.sum()),
                     "n_natural_isotopes": len(d),
                     "radioactive_isotopes": ",".join(f"{s}-{a} ({r})" for a, st, th, r in
                                                      zip(d.A, d.stable, d.t_half_s, d.ensdf_t_half_raw)
                                                      if not st and np.isfinite(th)),
                     "lower_limit_half_life": bool(d.t_half_limit.isin(limits).any()),
                     "unresolved": ",".join(f"{s}-{a}:{l}" for a, l, th in zip(d.A, d.t_half_limit, d.t_half_s)
                                            if not np.isfinite(th) and not np.isinf(th))})
    return pd.DataFrame(recs).sort_values("Z").reset_index(drop=True), iso


@lru_cache(maxsize=1)
def activities() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    u = c3.universe()
    els = sorted({e for lst in u["elements"] for e in lst})
    act, iso = element_activity(els)
    a_k = float(act.loc[act.element == "K", "A_Bq_per_g"].iloc[0])
    # strict: the Table B flag's elements, recovered from the data (every compound containing them is flagged)
    rad_rows = u["contains_radioactive"].to_numpy(bool)
    strict = {e for e in els if c3.core.contains(e).any() and rad_rows[c3.core.contains(e)].all()}
    act["strict"] = act.element.isin(strict)
    act["above_K"] = act.A_Bq_per_g > a_k
    act["moderate"] = act.strict | act.above_K
    act["named"] = act.strict | act.element.isin(NAMED)
    act["ratio_to_K"] = act.A_Bq_per_g / a_k
    return act, iso, a_k


def rad_mask(variant: str) -> np.ndarray:
    """Rows that SATISFY the RAD constraint under ``variant`` (strict | moderate | named)."""
    base = c3.constraint_mask("RAD")
    if variant == "strict":
        return base
    act, _, _ = activities()
    extra = act.loc[act[variant] & ~act.strict, "element"]
    m = base.copy()
    for e in extra:
        m &= ~c3.core.contains(e)
    return m
