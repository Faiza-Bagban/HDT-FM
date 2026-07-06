# models/temporal_transformer.py
"""
Temporal Transformer — processes a window of fused embeddings over time.
8-head, 2-layer, window T=30.

This is the HUD. Without it, the suit only sees "now."
With it, the suit sees patterns, trends, trajectories.

Tony Stark note: "JARVIS without memory is just Siri."
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=2,
                 dropout=0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(embed_dim, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, 256) — sequence of fused states over time window
        Returns:
            (B, 256) — temporally-aware representation (last token pooled)
        """
        x = self.pos_enc(x)
        x = self.encoder(x)          # (B, T, 256)
        x = self.norm(x[:, -1, :])    # (B, 256) — last step = most recent
        return x
