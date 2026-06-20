"""Lightweight 3D-CNN baselines for residual gas saturation prediction.

Inputs:
    voxel: 3D pore geometry tensor with shape (B, 1, D, D, D)
    features: Morphological descriptor tensor with shape (B, F)

Output:
    Predicted residual gas saturation score with shape (B,).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Conv3dBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleSrgNet(nn.Module):
    """Voxel CNN + feature MLP -> concat -> linear Srg head.

    Args:
        n_features: Dimensionality of numerical descriptor features.
        cnn_channels: Channel dimensions for the 3-layer CNN.
        feat_hidden: Hidden dimension of the feature MLP.
        head_hidden: Hidden dimension of the post-fusion head.
        dropout: Dropout rate for regularization (default: 0.1).
    """
    def __init__(
        self,
        n_features: int = 18,
        cnn_channels: tuple[int, ...] = (16, 32, 64),
        feat_hidden: int = 32,
        head_hidden: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        chs = (1,) + tuple(cnn_channels)
        self.cnn = nn.Sequential(*[Conv3dBlock(chs[i], chs[i + 1]) for i in range(len(cnn_channels))])
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.cnn_out_dim = cnn_channels[-1]

        self.feat_mlp = nn.Sequential(
            nn.Linear(n_features, feat_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_hidden, feat_hidden),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(self.cnn_out_dim + feat_hidden, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, voxel: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        h_v = self.cnn(voxel)
        h_v = self.global_pool(h_v).flatten(1)        # (B, C_last)
        h_f = self.feat_mlp(features)                  # (B, F)
        h = torch.cat([h_v, h_f], dim=1)
        out = self.head(h).squeeze(-1)                 # (B,) linear, without sigmoid
        return out


class SimpleSrgNetSigmoid(SimpleSrgNet):
    """SimpleSrgNet with sigmoid head."""
    def forward(self, voxel, features):
        return torch.sigmoid(super().forward(voxel, features))


class SimpleTauGateNet(SimpleSrgNet):
    """SimpleSrgNet with tau-guided channel gating."""

    def __init__(self, n_features: int = 18, tau_idx: int = 1, **kwargs) -> None:
        super().__init__(n_features=n_features, **kwargs)
        from models_3d import TauGate
        self.tau_gate = TauGate(self.cnn_out_dim, tau_idx=tau_idx)

    def forward(self, voxel, features):
        h_v = self.cnn(voxel)
        h_v = self.global_pool(h_v).flatten(1)
        h_v = self.tau_gate(h_v, features)             # tau-guided channel gating
        h_f = self.feat_mlp(features)
        h = torch.cat([h_v, h_f], dim=1)
        return torch.sigmoid(self.head(h).squeeze(-1))   # sigmoid head


class PhiOnlyBaseline(nn.Module):
    """Descriptor-only MLP baseline for comparing against voxel-fusion models."""

    def __init__(self, n_features: int = 18) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, voxel: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features).squeeze(-1)   # linear output, no sigmoid
