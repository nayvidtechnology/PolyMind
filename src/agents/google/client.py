from typing import Iterable
import os

from ..base.agent_base import AgentBase
from ..base.provider_registry import register_provider

try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None


class GoogleAgent(AgentBase):
    def __init__(self, model: str = "gemini-1.5-flash", api_key_env: str = "GOOGLE_API_KEY"):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in env: {api_key_env}")
        if genai is None:
            raise RuntimeError("google-generativeai not installed. See requirements.txt")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        # Simplified: send last user message
        user_text = "\n".join([m["content"] for m in messages if m["role"] == "user"]) or messages[-1]["content"]
        rsp = self.model.generate_content(user_text)
        return getattr(rsp, "text", "")

    def stream(self, messages: list[dict[str, str]], **kwargs) -> Iterable[str]:
        yield self.chat(messages, **kwargs)

    def name(self) -> str:
        return "google"


@register_provider("google")
def _factory(cfg: dict | None = None) -> AgentBase:
    if cfg:
        return GoogleAgent(model=cfg.get("model", "gemini-1.5-flash"), api_key_env=cfg.get("api_key_env", "GOOGLE_API_KEY"))
    return GoogleAgent()
