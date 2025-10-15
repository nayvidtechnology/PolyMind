from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import torch
import torch.nn as nn

from .registry import get_block


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    max_seq: int = 1024
    block: str = "transformer_block_v1"  # name in registry


class PolymindModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        Block = get_block(cfg.block)
        self.layers = nn.ModuleList([Block(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        # input_ids: [B, T]
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
