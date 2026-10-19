"""Load the tagged configuration (config/default.yaml) and resolve references into source records."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import yaml

from shieldmap import paths

TAGS = {"PHYS", "STD", "SWEEP"}
DEFAULT = paths.CONFIG / "default.yaml"


def _is_leaf(node: Any) -> bool:
    return isinstance(node, dict) and "tag" in node and ("value" in node or "ref" in node)


def iter_leaves(node: Any, prefix: str = "") -> Iterator[tuple[str, dict]]:
    """Yield (dotted.key, leaf) for every tagged parameter."""
    if _is_leaf(node):
        yield prefix, node
        return
    if isinstance(node, dict):
        for k, v in node.items():
            yield from iter_leaves(v, f"{prefix}.{k}" if prefix else str(k))
    else:
        raise ValueError(f"untagged config value at {prefix!r}: {node!r}")


@lru_cache(maxsize=4)
def raw(path: str | None = None) -> dict:
    with open(path or DEFAULT, encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(path: str | None = None) -> str:
    return hashlib.sha256(Path(path or DEFAULT).read_bytes()).hexdigest()


def resolve_ref(ref: str) -> Any:
    """Resolve 'sources/<file>.yaml#a.b.c' relative to config/."""
    file_part, _, key_path = ref.partition("#")
    with open(paths.CONFIG / file_part, encoding="utf-8") as f:
        node = yaml.safe_load(f)
    for k in filter(None, key_path.split(".")):
        node = node[k]
    return node


def leaf(key: str, path: str | None = None) -> dict:
    node = raw(path)
    for k in key.split("."):
        node = node[k]
    if not _is_leaf(node):
        raise KeyError(f"{key!r} is not a tagged parameter")
    return node


def get(key: str, path: str | None = None) -> Any:
    """Value of a tagged parameter (references are resolved)."""
    node = leaf(key, path)
    return resolve_ref(node["ref"]) if "ref" in node else node["value"]


def primary(key: str, path: str | None = None) -> Any:
    node = leaf(key, path)
    if node["tag"] != "SWEEP" or "primary" not in node:
        raise KeyError(f"{key!r} has no primary value")
    return node["primary"]


def validate(path: str | None = None, resolve: bool = True) -> list[str]:
    """Return a list of problems (empty if the configuration satisfies the parameter policy)."""
    problems = []
    for key, node in iter_leaves(raw(path)):
        tag = node.get("tag")
        if tag not in TAGS:
            problems.append(f"{key}: tag {tag!r} not in {sorted(TAGS)}")
        if not str(node.get("source", "")).strip():
            problems.append(f"{key}: missing source")
        if tag == "SWEEP":
            val = node.get("value")
            if not isinstance(val, list) or len(val) < 2:
                problems.append(f"{key}: SWEEP must be a list of >= 2 values")
            elif "primary" in node and node["primary"] not in val:
                problems.append(f"{key}: primary {node['primary']!r} not in sweep values")
        if tag == "STD":
            if "ref" in node:
                file_part = node["ref"].partition("#")[0]
                src = paths.CONFIG / file_part
                if not src.exists():
                    problems.append(f"{key}: referenced source {file_part} missing")
                    continue
                with open(src, encoding="utf-8") as f:
                    rec = yaml.safe_load(f)
                if not rec.get("citation"):
                    problems.append(f"{key}: source {file_part} has no citation")
                if resolve:
                    try:
                        resolve_ref(node["ref"])
                    except Exception as e:  # pragma: no cover - reported, not raised
                        problems.append(f"{key}: cannot resolve {node['ref']}: {e}")
            elif "sources/" not in str(node.get("source", "")) and "SRD" not in str(node.get("source", "")) \
                    and "pymatgen" not in str(node.get("source", "")):
                problems.append(f"{key}: STD value without a reference to a cited source")
    return problems
