from __future__ import annotations
from typing import Any, Dict


class AzureMLTrainer:
    def run(self, config: Dict[str, Any], dry_run: bool = True) -> None:
        print("[AzureMLTrainer] would submit job to Azure ML with config:")
        print({k: v for k, v in config.items() if k != "secrets"})
        if dry_run:
            print("[AzureMLTrainer] dry run complete.")
