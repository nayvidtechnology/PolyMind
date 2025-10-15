from __future__ import annotations
from typing import Any, Dict


class SageMakerTrainer:
    def run(self, config: Dict[str, Any], dry_run: bool = True) -> None:
        print("[SageMakerTrainer] would submit job to SageMaker with config:")
        print({k: v for k, v in config.items() if k != "secrets"})
        if dry_run:
            print("[SageMakerTrainer] dry run complete.")
