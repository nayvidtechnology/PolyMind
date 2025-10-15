from typing import Iterable, Optional
import os
from dataclasses import dataclass

from ..base.agent_base import AgentBase
from ..base.provider_registry import register_provider

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass
class OpenAIConfig:
    model: str = "gpt-4o-mini"
    endpoint: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"


class OpenAIAgent(AgentBase):
    def __init__(self, cfg: OpenAIConfig | None = None):
        cfg = cfg or OpenAIConfig()
        api_key = os.getenv(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in env: {cfg.api_key_env}")
        if OpenAI is None:  # type: ignore
            raise RuntimeError("openai package not installed. Install via requirements.txt")
        # openai>=1.0 style client
        self.client = OpenAI(api_key=api_key)  # type: ignore
        self.model = cfg.model

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        rsp = self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        return rsp.choices[0].message.content or ""

    def stream(self, messages: list[dict[str, str]], **kwargs) -> Iterable[str]:
        # Simple fallback: yield full text (non-streaming) to keep sample robust
        text = self.chat(messages, **kwargs)
        yield text

    def name(self) -> str:
        return "openai"


@register_provider("openai")
def _factory(cfg: dict | None = None) -> AgentBase:
    if cfg:
        ocfg = OpenAIConfig(
            model=cfg.get("model", "gpt-4o-mini"),
            endpoint=cfg.get("endpoint"),
            api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"),
        )
        return OpenAIAgent(ocfg)
    return OpenAIAgent()
