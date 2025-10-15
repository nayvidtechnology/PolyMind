from __future__ import annotations
from pathlib import Path
import yaml
from typing import Any, Dict


def load_yaml(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())
