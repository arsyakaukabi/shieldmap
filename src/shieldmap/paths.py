"""Project directory layout. All paths are absolute and derived from the package location."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # .../shieldmap
REPO_ROOT = ROOT.parent                              # .../shield (holds the .env file)
CONFIG = ROOT / "config"
SOURCES = CONFIG / "sources"
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports" / "phase1"
TESTS = ROOT / "tests"
FIXTURES = TESTS / "fixtures"


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
