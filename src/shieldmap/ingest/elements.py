"""Table A — elements Z = 1–100.

Fields: Z, symbol, atomic_mass (pymatgen, IUPAC), density_solid_exp (pymatgen ``density_of_solid``,
g/cm³, experimental), crustal_abundance_ppm (static fixture from the CRC Handbook table shipped by
``mendeleev``; see config/sources/crustal_abundance.yaml), is_radioactive, is_noble_gas.
Cross sections on the energy grid come from :mod:`shieldmap.ingest.xsec` (NIST XCOM 3.1).

Outputs (data/interim/elements/):
  elements.parquet   one row per element
  xsec_grid.npz      partial and total μ/ρ (cm²/g) on the common grid, shape (100, n_E)
  edges.parquet      absorption-edge energies (K, L1–L3, M1–M5, …) per element
"""
from __future__ import annotations

import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd

from shieldmap import config, paths
from shieldmap.ingest import fetch

OUT = paths.INTERIM / "elements"
ABUNDANCE_SOURCE = "crustal_abundance"
NO_STABLE_ISOTOPE_BELOW_84 = {43, 61}          # Tc, Pm


def is_radioactive(z: int) -> bool:
    """No stable isotope: Tc, Pm and every Z >= 84 (Bi-209 treated as stable)."""
    return z in NO_STABLE_ISOTOPE_BELOW_84 or z >= 84


def make_abundance_fixture() -> dict:
    """Freeze the CRC crustal-abundance column (as shipped by mendeleev) into a source record."""
    from mendeleev.fetch import fetch_table
    import mendeleev

    tab = fetch_table("elements")[["atomic_number", "symbol", "abundance_crust"]]
    tab = tab[tab.atomic_number <= config.get("elements.z_max")]
    meta = fetch_table("propertymetadata")
    row = meta[(meta.table_name == "elements") & (meta.column_name == "abundance_crust")].iloc[0]
    from pathlib import Path
    db = Path(mendeleev.__file__).parent / "elements.db"
    values = {str(r.symbol): (None if pd.isna(r.abundance_crust) else float(r.abundance_crust))
              for r in tab.itertuples()}
    record = {
        "title": "Abundance of elements in the Earth's crust (mg/kg = ppm by mass)",
        "citation": ("W. M. Haynes (ed.), CRC Handbook of Chemistry and Physics, 95th ed., CRC Press, "
                     "Boca Raton (2014), section 14, 'Abundance of Elements in the Earth's Crust and "
                     "in the Sea' (crust column), as distributed in the mendeleev package "
                     f"(citation key {row.citation_keys})."),
        "data_class": "P3",
        "retrieval_method": "official package (PyPI mendeleev), frozen once into this fixture",
        "package": {"name": "mendeleev", "version": mendeleev.__version__,
                    "elements_db_sha256": fetch.sha256(db)},
        "retrieved_utc": fetch.utc_now(),
        "unit": str(row.unit),
        "licence": "mendeleev: MIT; values: CRC Handbook (factual data, cited)",
        "notes": "Elements without a tabulated value are null; nothing is imputed.",
        "data": {"abundance_ppm": values},
    }
    fetch.write_source(ABUNDANCE_SOURCE, record)
    return record


def abundance_table() -> dict:
    rec = fetch.read_source(ABUNDANCE_SOURCE) or make_abundance_fixture()
    return rec["data"]["abundance_ppm"]


def build_metadata() -> pd.DataFrame:
    from pymatgen.core import Element

    abundance = abundance_table()
    noble = set(config.get("elements.noble_gases"))
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for z in range(1, config.get("elements.z_max") + 1):
            el = Element.from_Z(z)
            d = el.density_of_solid                       # kg m^-3, experimental; None if absent
            if d is not None:
                assert str(d.unit) == "kg m^-3", d.unit
            rows.append({
                "Z": z,
                "symbol": el.symbol,
                "atomic_mass": float(el.atomic_mass),
                "density_solid_exp": None if d is None else float(d) / 1000.0,   # g/cm³
                "crustal_abundance_ppm": abundance.get(el.symbol),
                "is_radioactive": is_radioactive(z),
                "is_noble_gas": el.symbol in noble,
            })
    df = pd.DataFrame(rows)
    df["density_solid_exp"] = df["density_solid_exp"].astype("float64")
    df["crustal_abundance_ppm"] = df["crustal_abundance_ppm"].astype("float64")
    return df


def write_pymatgen_source() -> None:
    import pymatgen.core
    from pathlib import Path
    pt = Path(pymatgen.core.__file__).parent / "periodic_table.json.gz"
    if not pt.exists():
        pt = Path(pymatgen.core.__file__).parent / "periodic_table.json"
    prior = fetch.read_source("pymatgen_elements") or {}
    old_sha = prior.get("package", {}).get("periodic_table_sha256")
    same = old_sha == fetch.sha256(pt)
    if old_sha and not same:
        raise ValueError("pymatgen element table changed since it was recorded; review the change and "
                         "delete config/sources/pymatgen_elements.yaml to accept it")
    try:
        from importlib.metadata import packages_distributions
        owner = packages_distributions().get("pymatgen", ["pymatgen"])
        owners = {d: version(d) for d in owner}
    except Exception:  # pragma: no cover
        owners = {"pymatgen": version("pymatgen")}
    fetch.write_source("pymatgen_elements", {
        "title": "pymatgen Element data: atomic mass and experimental density of the solid",
        "citation": ("S. P. Ong et al., Python Materials Genomics (pymatgen): a robust, open-source "
                     "python library for materials analysis, Comput. Mater. Sci. 68, 314 (2013). "
                     "Element.atomic_mass (IUPAC standard atomic weights) and Element.density_of_solid "
                     "(experimental solid density, kg m^-3)."),
        "data_class": "P1",
        "retrieval_method": "official package (PyPI pymatgen)",
        "package": {"name": "pymatgen", "version": version("pymatgen"), "distributions": owners,
                    "periodic_table_file": pt.name, "periodic_table_sha256": fetch.sha256(pt)},
        "retrieved_utc": prior["retrieved_utc"] if same and prior.get("retrieved_utc") else fetch.utc_now(),
        "licence": "MIT",
    })


def build(write: bool = True) -> pd.DataFrame:
    """Build Table A (metadata + cross sections on the grid)."""
    from shieldmap.ingest import xsec

    df = build_metadata()
    if write:
        paths.ensure(OUT)
        write_pymatgen_source()
        df.to_parquet(OUT / "elements.parquet", index=False)
        xsec.build()
    return df


if __name__ == "__main__":
    build()
