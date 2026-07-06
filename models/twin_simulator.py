# models/twin_simulator.py
"""
Digital Twin Simulator — 2-layer MLP.
Predicts next health state vector from current temporal context.
Trained with self-supervised loss (predict next fused state).

This is literally the thing the paper is named after.
Without this, HDT-FM is just... HFM. Which isn't a thing.

Tony Stark note: "The suit doesn't just react. It predicts."
"""

import torch
import torch.nn as nn


class TwinSimulator(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 256) temporal context from TemporalTransformer
        Returns:
            next_state: (B, 256) predicted next health state
        """
        return self.mlp(x)
