from __future__ import annotations
from typing import Any, Dict


class VertexTrainer:
    def run(self, config: Dict[str, Any], dry_run: bool = True) -> None:
        print("[VertexTrainer] would submit job to Vertex AI with config:")
        print({k: v for k, v in config.items() if k != "secrets"})
        if dry_run:
            print("[VertexTrainer] dry run complete.")
