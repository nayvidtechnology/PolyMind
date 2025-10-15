"""
PolyMind: Modular Multi-Modal LLM Framework

A compact, modular multi-modal LLM stack with pluggable components.
"""

__version__ = "1.1.0"

# Core model and registry
from .core import PolymindModel, register_block, get_block
from .trainers import get_trainer

# Agent system
from .agents.base import AgentBase, register_provider, get_provider

# Storage adapters
from .storage import Storage, LocalFS

__all__ = [
    "__version__",
    "PolymindModel",
    "register_block",
    "get_block",
    "get_trainer",
    "AgentBase",
    "register_provider",
    "get_provider",
    "Storage",
    "LocalFS",
]
