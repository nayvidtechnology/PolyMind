from typing import Iterable, Optional
import os

from ..base.agent_base import AgentBase
from ..base.provider_registry import register_provider

try:
    from azure.ai.inference import ChatCompletionsClient  # type: ignore
    from azure.core.credentials import AzureKeyCredential  # type: ignore
except Exception:  # pragma: no cover
    ChatCompletionsClient = None  # type: ignore
    AzureKeyCredential = None  # type: ignore


class AzureAgent(AgentBase):
    def __init__(self, endpoint: Optional[str] = None, deployment: str = "gpt-4o-mini", api_key_env: str = "AZURE_OPENAI_API_KEY"):
        endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv(api_key_env)
        if not endpoint or not api_key:
            raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT or API key env")
        if ChatCompletionsClient is None or AzureKeyCredential is None:
            raise RuntimeError("azure-ai-inference not installed. See requirements.txt")
        self.client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
        self.deployment = deployment

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        rsp = self.client.complete(self.deployment, messages=messages)
        return getattr(rsp, "choices", [{}])[0].get("message", {}).get("content", "")

    def stream(self, messages: list[dict[str, str]], **kwargs) -> Iterable[str]:
        yield self.chat(messages, **kwargs)

    def name(self) -> str:
        return "azure"


@register_provider("azure")
def _factory(cfg: dict | None = None) -> AgentBase:
    if cfg:
        return AzureAgent(
            endpoint=cfg.get("endpoint"),
            deployment=cfg.get("deployment", "gpt-4o-mini"),
            api_key_env=cfg.get("api_key_env", "AZURE_OPENAI_API_KEY"),
        )
    return AzureAgent()
