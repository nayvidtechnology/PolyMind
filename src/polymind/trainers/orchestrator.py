from __future__ import annotations
from typing import Protocol, Dict, Any


class Trainer(Protocol):
    def run(self, config: Dict[str, Any], dry_run: bool = True) -> None: ...


def get_trainer(name: str) -> Trainer:
    key = name.lower()
    if key == "local":
        from .runners.local import LocalTrainer
        return LocalTrainer()
    if key == "azureml":
        from .runners.azureml import AzureMLTrainer
        return AzureMLTrainer()
    if key == "vertex":
        from .runners.vertex import VertexTrainer
        return VertexTrainer()
    if key == "sagemaker":
        from .runners.sagemaker import SageMakerTrainer
        return SageMakerTrainer()
    raise ValueError(f"Unknown trainer backend: {name}")
