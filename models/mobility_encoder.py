# models/mobility_encoder.py
"""
Mobility Encoder — Bidirectional GRU.
Takes synthetic GPS traces (B, T, 4) → learned 64-d representation.
Replaces torch.zeros(64) aka "the modality that didn't exist."

Tony Stark note: "Calling zeros a modality is like calling
an empty suit Iron Man."
"""

import torch
import torch.nn as nn


class MobilityEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, num_layers=2,
                 out_dim=64, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        gru_out = hidden_dim * 2  # bidirectional
        self.out_proj = nn.Sequential(
            nn.LayerNorm(gru_out),
            nn.Linear(gru_out, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, 4) GPS traces — [lat, lon, speed, heading]
        Returns:
            (B, 64) learned mobility embedding
        """
        out, _ = self.gru(x)            # (B, T, hidden*2)
        pooled = out.mean(dim=1)         # (B, hidden*2)
        return self.out_proj(pooled)     # (B, 64)
