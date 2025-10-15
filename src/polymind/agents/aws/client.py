from typing import Iterable
import os

from ..base.agent_base import AgentBase
from ..base.provider_registry import register_provider

try:
    import boto3  # type: ignore
except Exception:  # pragma: no cover
    boto3 = None


class AWSAgent(AgentBase):
    def __init__(self, model: str = "anthropic.claude-3-haiku-20240307-v1:0", region: str | None = None):
        if boto3 is None:
            raise RuntimeError("boto3 not installed. See requirements.txt")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.client = boto3.client("bedrock-runtime", region_name=self.region)
        self.model = model

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        # Simplified Anthropic-style prompt
        user_text = "\n".join([m["content"] for m in messages if m["role"] == "user"]) or messages[-1]["content"]
        body = {
            "prompt": f"\n\nHuman: {user_text}\n\nAssistant:",
            "max_tokens_to_sample": 256,
        }
        rsp = self.client.invoke_model(modelId=self.model, body=body)
        out = rsp.get("body", {}).get("completion") if isinstance(rsp.get("body"), dict) else None
        return out or ""

    def stream(self, messages: list[dict[str, str]], **kwargs) -> Iterable[str]:
        yield self.chat(messages, **kwargs)

    def name(self) -> str:
        return "aws"


@register_provider("aws")
def _factory(cfg: dict | None = None) -> AgentBase:
    if cfg:
        return AWSAgent(model=cfg.get("model", "anthropic.claude-3-haiku-20240307-v1:0"), region=cfg.get("region"))
    return AWSAgent()
