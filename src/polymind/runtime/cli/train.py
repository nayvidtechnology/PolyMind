import click  # type: ignore
import yaml  # type: ignore
from pathlib import Path
from typing import Any, Dict

from polymind.trainers import get_trainer  # type: ignore


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--dry-run/--no-dry-run", default=True)
def main(config: str, dry_run: bool):
    cfg: Dict[str, Any] = yaml.safe_load(Path(config).read_text())
    print("[train] loaded config:")
    print(cfg)
    backend = cfg.get("trainer", "local")
    trainer = get_trainer(backend)
    trainer.run(cfg, dry_run=dry_run)


if __name__ == "__main__":
    main()  # type: ignore[misc]
