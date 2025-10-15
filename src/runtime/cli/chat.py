import click  # type: ignore
from typing import List, Any, cast
from pathlib import Path
import os
import yaml  # type: ignore

from agents.base.provider_registry import get_provider  # type: ignore


@click.command()
@click.option("--provider", required=True, help="Provider name: openai|google|azure|aws|local")
@click.option("--message", required=True, help="User message")
@click.option("--system", default="You are a helpful assistant.")
@click.option("--stream/--no-stream", default=False)
@click.option("--config", "config_path", default=None, help="Path to provider YAML config; defaults to configs/providers/{provider}.yaml if present")
def main(provider: str, message: str, system: str, stream: bool, config_path: str | None):
    cfg: dict[str, Any] | None = None
    # Determine config path
    if config_path:
        cfg_file = Path(config_path)
    else:
        # Default to repository_root/configs/providers/{provider}.yaml if exists
        repo_root = Path(__file__).resolve().parents[3]
        cfg_file = repo_root / "configs" / "providers" / f"{provider}.yaml"
    # Load YAML if available
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            raw = loaded if isinstance(loaded, dict) else {}

        def _expand(obj):
            if isinstance(obj, dict):
                return {k: _expand(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_expand(v) for v in obj]
            if isinstance(obj, str):
                return os.path.expandvars(obj)
            return obj

    cfg = cast(dict[str, Any], _expand(raw))

    agent = get_provider(provider, cfg)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": message},
    ]
    if stream:
        for token in agent.stream(messages):
            print(token, end="", flush=True)
        print()
    else:
        text = agent.chat(messages)
        print(text)


if __name__ == "__main__":
    # Click handles args at runtime; static analyzers may warn about missing params
    main()  # type: ignore[misc]
