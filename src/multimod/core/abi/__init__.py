from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import warnings

try:
    import jsonschema  # type: ignore
except Exception as e:  # pragma: no cover
    jsonschema = None  # type: ignore


def _load_schema() -> Dict[str, Any]:
    schema_path = Path(__file__).parent / "schema" / "mm_abi.json"
    import json

    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(meta)
    proj = out.get("projector") or {}
    if "T_cap" in proj and "t_cap" not in proj:
        proj = dict(proj)
        proj["t_cap"] = proj["T_cap"]
        warnings.warn("MM-ABI: 'T_cap' is deprecated; use 't_cap' instead.", DeprecationWarning)
        out["projector"] = proj
    return out


def validate_mm_abi(meta: Dict[str, Any]) -> Dict[str, Any]:
    norm = normalize_metadata(meta)
    if jsonschema is None:
        return norm
    schema = _load_schema()
    jsonschema.validate(instance=norm, schema=schema)  # type: ignore[attr-defined]
    return norm
