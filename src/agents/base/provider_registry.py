from typing import Callable, Dict, Any
import importlib

from .agent_base import AgentBase


_REGISTRY: Dict[str, Callable[[dict | None], AgentBase]] = {}


def register_provider(name: str):
    def _wrap(factory: Callable[..., AgentBase]):
        # Wrap to accept optional config dict for backward compatibility
        def wrapper(cfg: dict | None = None) -> AgentBase:
            try:
                return factory(cfg)  # type: ignore[misc]
            except TypeError:
                return factory()  # type: ignore[misc]

        _REGISTRY[name.lower()] = wrapper
        return factory

    return _wrap


def _attempt_autoload(key: str) -> None:
    try:
        importlib.import_module(f"agents.{key}.client")
    except ModuleNotFoundError:
        pass


def get_provider(name: str, cfg: dict | None = None) -> AgentBase:
    key = name.lower()
    if key not in _REGISTRY:
        _attempt_autoload(key)
    if key not in _REGISTRY:
        raise ValueError(f"Provider '{name}' not registered. Available: {list(_REGISTRY)}")
    return _REGISTRY[key](cfg)
