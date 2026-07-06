# models/twin_simulator.py
"""
Digital Twin Simulator — 2-layer MLP.
Two heads:
  1. State predictor:  (B, 256) → (B, 256) next health state
  2. Reconstructor:    (B, 256) → (B, 20)  reconstruct input HRV features

Self-supervised = reconstruct the activity features you were given.
No more predicting random strangers' embeddings.

Tony Stark note: "The suit doesn't just react. It *understands* the pilot."
"""

import torch
import torch.nn as nn


class TwinSimulator(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=512, recon_dim=20, dropout=0.1):
        super().__init__()
        # State prediction head (for inference / paper narrative)
        self.state_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        # Reconstruction head (self-supervised training signal)
        self.recon_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, recon_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 256) temporal context
        Returns:
            next_state: (B, 256) predicted next state
            recon:      (B, 20) reconstructed activity features
        """
        return self.state_head(x), self.recon_head(x)
