from typing import Iterable

from ..base.agent_base import AgentBase
from ..base.provider_registry import register_provider


class LocalEchoAgent(AgentBase):
    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"[local-echo] {last}"

    def stream(self, messages: list[dict[str, str]], **kwargs) -> Iterable[str]:
        yield self.chat(messages, **kwargs)

    def name(self) -> str:
        return "local"


@register_provider("local")
def _factory():
    return LocalEchoAgent()
