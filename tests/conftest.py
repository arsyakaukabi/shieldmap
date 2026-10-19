"""Skip, rather than fail, the tests whose inputs are not part of this repository.

Not shipped (licence audit, README "What is in data/"): raw upstream inputs under data/raw/, intermediates that only
`python -m shieldmap.build` or the analysis stages rebuild (most of data/interim/, data/processed/, data/phase2-4d/),
the NIST SRD 126 fixtures and air table (scripts/restore_nist_srd126.py), the per-element crustal abundances
(scripts/restore_crustal_abundance.py) and files outside the repository. A test that stops on one of these is
reported as skipped with the missing input named; every other failure is reported as a failure.
"""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SRD126 = ROOT / "config" / "sources" / "nist_srd126.yaml"
CRUSTAL = ROOT / "config" / "sources" / "crustal_abundance.yaml"


def _block_missing(path: Path, key: str) -> bool:
    try:
        return key not in (yaml.safe_load(path.read_text(encoding="utf-8")).get("data") or {})
    except OSError:
        return True


SRD126_ABSENT = _block_missing(SRD126, "air")
CRUSTAL_ABSENT = _block_missing(CRUSTAL, "abundance_ppm")


NOT_SHIPPED = [ROOT / "data", ROOT / "reports", ROOT / "config" / "phase4a"]


def _reason(exc) -> str | None:
    if exc.errisinstance(FileNotFoundError):
        missing = getattr(exc.value, "filename", None)
        if missing:
            p = Path(str(missing)).resolve()
            if ROOT not in p.parents or any(p == d or d in p.parents for d in NOT_SHIPPED) \
                    or p.name.startswith("srd126_"):
                return f"input not in this repository (see README): {missing}"
        return None
    text = str(exc.value)
    stripped = exc.errisinstance(KeyError) and text in ("'data'", "'air'", "'abundance_ppm'")
    if SRD126_ABSENT and (stripped or (exc.errisinstance(AssertionError) and (
            "nist_srd126.yaml#data.air" in text or "nist_srd126: missing data" in text))):
        return "NIST SRD 126 air table not restored (python scripts/restore_nist_srd126.py)"
    if CRUSTAL_ABSENT and (stripped or (exc.errisinstance(AssertionError) and (
            "crustal_abundance.yaml#" in text or "crustal_abundance: missing data" in text))):
        return "crustal abundances not restored (python scripts/restore_crustal_abundance.py)"
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    exc = call.excinfo
    if rep.failed and exc is not None:
        why = _reason(exc)
        if why:
            rep.outcome = "skipped"
            rep.longrepr = (str(item.path), item.location[1] or 0, f"Skipped: {why}")
