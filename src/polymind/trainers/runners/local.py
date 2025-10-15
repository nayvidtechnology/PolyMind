from __future__ import annotations
from typing import Any, Dict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from polymind.core.model import PolymindModel, ModelConfig
from polymind.storage.local_fs import LocalFS


class LocalTrainer:
    def _build_model(self, cfg: Dict[str, Any]) -> nn.Module:
        mcfg = cfg.get("model", {})
        mc = ModelConfig(
            vocab_size=int(mcfg.get("vocab_size", 32000)),
            d_model=int(mcfg.get("d_model", 512)),
            n_layers=int(mcfg.get("n_layers", 6)),
            n_heads=int(mcfg.get("n_heads", 8)),
            max_seq=int(mcfg.get("max_seq", 1024)),
            block=str(mcfg.get("block", "transformer_block_v1")),
        )
        return PolymindModel(mc)

    def run(self, config: Dict[str, Any], dry_run: bool = True) -> None:
        print("[LocalTrainer] config:", {k: v for k, v in config.items() if k != "secrets"})
        if dry_run:
            print("[LocalTrainer] dry run complete.")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LocalTrainer] using device: {device}")

        # Build model
        model = self._build_model(config).to(device)
        model.train()

        # Synthetic batch
        B = 2
        T = min(16, int(config.get("model", {}).get("max_seq", 1024)))
        vocab = int(config.get("model", {}).get("vocab_size", 32000))
        inputs = torch.randint(0, vocab, (B, T), device=device)
        targets = torch.randint(0, vocab, (B, T), device=device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        # One step
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)  # [B, T, V]
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        print(f"[LocalTrainer] step complete. loss={loss.item():.4f}")

        # Save checkpoint locally
        ckpt_dir = Path(config.get("checkpoints", {}).get("local_dir", "models/checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        out_file = ckpt_dir / "polymind_one_step.pt"
        tmp = ckpt_dir / "_tmp_polymind_one_step.pt"
        torch.save({"model": model.state_dict(), "config": config}, tmp)
        LocalFS().save(tmp, str(out_file))
        print(f"[LocalTrainer] checkpoint saved: {out_file}")
