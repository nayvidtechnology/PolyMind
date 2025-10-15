from __future__ import annotations
from typing import Callable, Dict, Any

_BLOCKS: Dict[str, Callable[..., Any]] = {}


def register_block(name: str):
    def deco(fn: Callable[..., Any]):
        key = name.lower()
        if key in _BLOCKS:
            raise ValueError(f"Block '{name}' already registered")
        _BLOCKS[key] = fn
        return fn
    return deco


def get_block(name: str) -> Callable[..., Any]:
    key = name.lower()
    if key not in _BLOCKS:
        raise KeyError(f"Block '{name}' not found. Available: {list(_BLOCKS)}")
    return _BLOCKS[key]
