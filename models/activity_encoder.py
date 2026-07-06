# models/activity_encoder.py
"""
Sleep/Activity Encoder — MLP with residual connection.
Takes raw HRV features (20-d from WESAD CSV) → learned 128-d representation.
Replaces the zero-padded hack.

Tony Stark note: "A spreadsheet is not an encoder, kid."
"""

import torch
import torch.nn as nn


class ActivityEncoder(nn.Module):
    def __init__(self, in_dim=20, hidden_dim=64, out_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (B, in_dim) raw HRV features from WESAD CSV
        Returns:
            (B, 128) learned activity embedding
        """
        return self.net(x)
