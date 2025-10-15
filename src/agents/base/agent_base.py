from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any


class AgentBase(ABC):
    """Abstract base for chat/inference agents across providers."""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Synchronous chat completion.
        messages: [{"role": "user|system|assistant", "content": str}]
        """
        raise NotImplementedError

    @abstractmethod
    def stream(self, messages: list[dict[str, str]], **kwargs: Any) -> Iterable[str]:
        """Token streaming for chat completion."""
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        return self.__class__.__name__
