"""Helpers shared by the scripts in this folder."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def splice_yaml(rel: str, block: dict, note_lines: int = 2) -> bytes:
    """The shipped YAML without its stripped data block (last two comment lines) plus the restored block, dumped
    exactly as shieldmap.ingest.fetch.write_source dumps a source record."""
    import yaml
    text = (ROOT / rel).read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    if len(lines) >= note_lines and all(ln.startswith("# ") for ln in lines[-note_lines:]):
        text = "".join(lines[:-note_lines])
    elif "\ndata:\n" in text:
        text = text[:text.rfind("\ndata:\n") + 1]
    return (text + yaml.safe_dump({"data": block}, sort_keys=False, allow_unicode=True, width=100)).encode("utf-8")
