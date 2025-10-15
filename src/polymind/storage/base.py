from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO


class Storage(ABC):
    @abstractmethod
    def save(self, src: Path, dest: str) -> None: ...

    @abstractmethod
    def load(self, src: str, dest: Path) -> None: ...
