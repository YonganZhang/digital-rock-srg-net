"""3D backbones, fusion modules, and model-zoo components for Srg prediction.

Design principles:
- Small-sample friendly: compact 3D CNN/ResNet variants for the 360-sample setting.
- Range-consistent Srg outputs through sigmoid heads or logit-space training.
- Swappable fusion modules: concat, cross-attention, FiLM, and TauGate variants.

References:
- He et al. 2016 (ResNet): residual basic blocks.
- Perez et al. 2018 (FiLM): feature-wise linear modulation.
- Hu et al. 2018 (SE): channel attention.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Lightweight 3D ResNet components
# =========================================================================

def conv3x3x3(c_in: int, c_out: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = conv3x3x3(c_in, c_out, stride)
        self.bn1 = nn.BatchNorm3d(c_out)
        self.conv2 = conv3x3x3(c_out, c_out)
        self.bn2 = nn.BatchNorm3d(c_out)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv3d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(c_out),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNet3D(nn.Module):
    """Lightweight 3D ResNet returning pooled features with shape (B, C_last)."""

    def __init__(
        self,
        in_channels: int = 1,
        base_width: int = 32,
        layers: tuple[int, ...] = (2, 2, 2, 2),
        global_pool: bool = True,
    ) -> None:
        super().__init__()
        self.base_width = base_width
        self.global_pool = global_pool

        widths = [base_width * (2 ** i) for i in range(len(layers))]   # 32, 64, 128, 256

        # Stem: 128^3 -> 64^3.
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm3d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),    # 64^3 -> 32^3.
        )

        # Stages: 32^3 -> 32^3 -> 16^3 -> 8^3 -> 4^3.
        self.stages = nn.ModuleList()
        c_prev = base_width
        for i, (n_blocks, w) in enumerate(zip(layers, widths)):
            stride_first = 1 if i == 0 else 2
            blocks = []
            blocks.append(BasicBlock3D(c_prev, w, stride=stride_first))
            for _ in range(n_blocks - 1):
                blocks.append(BasicBlock3D(w, w, stride=1))
            self.stages.append(nn.Sequential(*blocks))
            c_prev = w

        self.out_channels = c_prev
        self.gap = nn.AdaptiveAvgPool3d(1) if global_pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        if self.global_pool:
            x = self.gap(x).flatten(1)
        return x


# =========================================================================
# Fusion modules: Concat / Cross-Attn / FiLM
# =========================================================================

class ConcatFusion(nn.Module):
    """Concatenation fusion baseline."""

    def __init__(self, voxel_dim: int, feat_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(voxel_dim + feat_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, h_v: torch.Tensor, h_f: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([h_v, h_f], dim=1))


class CrossAttnFusion(nn.Module):
    """Feature-query and voxel-key/value fusion.

    h_f -> query token with shape (B, 1, D)
    h_v -> key/value tokens with shape (B, 1, D)

    The voxel pathway is globally pooled before fusion, so this module implements
    channel-wise feature-conditioned weighting rather than dense spatial attention.
    """

    def __init__(self, voxel_dim: int, feat_dim: int, out_dim: int, heads: int = 4) -> None:
        super().__init__()
        # Project both pathways into a shared embedding dimension.
        self.embed_dim = max(voxel_dim, feat_dim)
        self.q_proj = nn.Linear(feat_dim, self.embed_dim)
        self.k_proj = nn.Linear(voxel_dim, self.embed_dim)
        self.v_proj = nn.Linear(voxel_dim, self.embed_dim)
        self.heads = heads
        self.out_proj = nn.Linear(self.embed_dim, out_dim)
        self.feat_residual = nn.Linear(feat_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, h_v: torch.Tensor, h_f: torch.Tensor) -> torch.Tensor:
        # (B, embed_dim)
        q = self.q_proj(h_f)
        k = self.k_proj(h_v)
        v = self.v_proj(h_v)
        # Single-token scaled dot-product attention.
        scale = self.embed_dim ** -0.5
        attn = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) * scale)   # (B, 1)
        # Gate value features with the attention weight and add a feature residual.
        attended = attn * v
        out = self.out_proj(F.relu(attended)) + self.feat_residual(h_f)
        return out


class FiLMFusion(nn.Module):
    """FiLM fusion (Perez et al. 2018): features generate gamma and beta.

    h_v_modulated = gamma * h_v + beta
    """

    def __init__(self, voxel_dim: int, feat_dim: int, out_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, voxel_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, voxel_dim),
        )
        self.fc = nn.Sequential(
            nn.Linear(voxel_dim + feat_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, h_v: torch.Tensor, h_f: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_net(h_f)
        beta = self.beta_net(h_f)
        h_v_mod = gamma * h_v + beta
        return self.fc(torch.cat([h_v_mod, h_f], dim=1))


# =========================================================================
# Model registry
# =========================================================================

FUSION_REGISTRY = {
    "concat": ConcatFusion,
    "cross_attn": CrossAttnFusion,
    "film": FiLMFusion,
}


# =========================================================================
# Lightweight backbone library
# =========================================================================

class ResNet10Tiny(nn.Module):
    """ResNet10-tiny: base=16, layers=(1,1,1,1), ~500K params.

    Compact ResNet-style baseline for the small-sample setting.
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        cnn = ResNet3D(in_channels=in_channels, base_width=16, layers=(1, 1, 1, 1))
        self.backbone = cnn
        self.out_channels = cnn.out_channels   # 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class InceptionBlock3D(nn.Module):
    """3D Inception block for MS-PoreNet with parallel 1x1x1, 3x3x3, and 5x5x5 branches."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        c_branch = c_out // 4
        self.b1 = nn.Sequential(
            nn.Conv3d(c_in, c_branch, kernel_size=1, bias=False),
            nn.BatchNorm3d(c_branch),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv3d(c_in, c_branch, kernel_size=1, bias=False),
            nn.BatchNorm3d(c_branch),
            nn.ReLU(inplace=True),
            nn.Conv3d(c_branch, c_branch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(c_branch),
            nn.ReLU(inplace=True),
        )
        self.b5 = nn.Sequential(
            nn.Conv3d(c_in, c_branch, kernel_size=1, bias=False),
            nn.BatchNorm3d(c_branch),
            nn.ReLU(inplace=True),
            nn.Conv3d(c_branch, c_branch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm3d(c_branch),
            nn.ReLU(inplace=True),
        )
        self.bp = nn.Sequential(
            nn.MaxPool3d(3, stride=1, padding=1),
            nn.Conv3d(c_in, c_branch, kernel_size=1, bias=False),
            nn.BatchNorm3d(c_branch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1)


class MSPoreNet(nn.Module):
    """MS-PoreNet: multi-scale 3D Inception backbone, ~800K params.

    Stem: 5x5x5 stride 2 -> 64^3
    Inception block x 3 with max-pooling downsampling: 64^3 -> 32^3 -> 16^3 -> 8^3
    Global average pooling returns C-dimensional features.
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),   # -> 32^3
        )
        self.inc1 = InceptionBlock3D(24, 64)    # 64 ch
        self.pool1 = nn.MaxPool3d(2)            # -> 16^3
        self.inc2 = InceptionBlock3D(64, 96)    # 96 ch
        self.pool2 = nn.MaxPool3d(2)            # -> 8^3
        self.inc3 = InceptionBlock3D(96, 128)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.out_channels = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.inc1(x); x = self.pool1(x)
        x = self.inc2(x); x = self.pool2(x)
        x = self.inc3(x)
        return self.gap(x).flatten(1)


class MBConvBlock3D(nn.Module):
    """MBConv block (depthwise separable convolution + SE) for CoAtNet-style stages."""

    def __init__(self, c_in: int, c_out: int, expansion: int = 4, stride: int = 1) -> None:
        super().__init__()
        c_hidden = c_in * expansion
        self.use_residual = (stride == 1 and c_in == c_out)
        layers = []
        if expansion != 1:
            layers += [nn.Conv3d(c_in, c_hidden, 1, bias=False),
                       nn.BatchNorm3d(c_hidden), nn.GELU()]
        layers += [
            nn.Conv3d(c_hidden, c_hidden, 3, stride=stride, padding=1, groups=c_hidden, bias=False),
            nn.BatchNorm3d(c_hidden), nn.GELU(),
            # SE
            nn.AdaptiveAvgPool3d(1),     # squeeze step; SE projection uses 1x1x1 convolutions
        ]
        self.proj1 = nn.Sequential(*layers[:-1])    # excludes GAP
        self.se_fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(c_hidden, c_hidden // 4, 1),
            nn.GELU(),
            nn.Conv3d(c_hidden // 4, c_hidden, 1),
            nn.Sigmoid(),
        )
        self.proj2 = nn.Sequential(
            nn.Conv3d(c_hidden, c_out, 1, bias=False),
            nn.BatchNorm3d(c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj1(x)
        h = h * self.se_fc(h)
        h = self.proj2(h)
        if self.use_residual:
            h = h + x
        return h


class TransformerEncoderLayer3D(nn.Module):
    """Standard Transformer encoder block used by later PoreFormer/PoreCoAt stages."""

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C), where N is the number of tokens.
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class PoreCoAt(nn.Module):
    """PoreCoAt: compact CoAtNet-style backbone with 2 conv stages and 2 attention stages.

    Stem: stride-2 conv -> 64^3
    Stage 1: MBConv x 2, 64^3 -> 32^3
    Stage 2: MBConv x 2, 32^3 -> 16^3
    Stage 3: Transformer encoder x 2 after downsampling to 8^3 tokens
    Stage 4: Transformer encoder x 2
    Global average pooling returns C-dimensional features.
    """

    def __init__(self, in_channels: int = 1, dim: int = 96) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(24), nn.GELU(),                # 64^3
            nn.MaxPool3d(2),                              # 32^3
        )
        self.s1 = nn.Sequential(MBConvBlock3D(24, 48, stride=2),
                                MBConvBlock3D(48, 48))     # 16^3
        self.s2 = nn.Sequential(MBConvBlock3D(48, dim, stride=2),
                                MBConvBlock3D(dim, dim))   # 8^3 = 512 tokens
        # tokens for transformer
        self.attn_layers = nn.ModuleList([
            TransformerEncoderLayer3D(dim, heads=4, mlp_ratio=2.0, dropout=0.1) for _ in range(2)
        ])
        self.norm = nn.LayerNorm(dim)
        self.out_channels = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)                  # (B, dim, 8, 8, 8)
        B, C, D1, D2, D3 = x.shape
        tokens = x.flatten(2).transpose(1, 2)   # (B, 512, dim)
        for layer in self.attn_layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        return tokens.mean(dim=1)       # GAP over tokens -> (B, dim)


class PoreFormer(nn.Module):
    """PoreFormer: Conv stem + Transformer encoder x 4, ~1.5M params.
    """

    def __init__(self, in_channels: int = 1, dim: int = 128, depth: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=4, stride=4, padding=0, bias=False),  # 32^3
            nn.BatchNorm3d(32), nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=2, stride=2, padding=0, bias=False),           # 16^3
            nn.BatchNorm3d(64), nn.GELU(),
            nn.Conv3d(64, dim, kernel_size=2, stride=2, padding=0, bias=False),          # 8^3
            nn.BatchNorm3d(dim),
        )
        # Learnable positional embedding for 8^3 = 512 tokens.
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer3D(dim, heads=4, mlp_ratio=2.0, dropout=0.1) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.out_channels = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        B, C, D, _, _ = x.shape
        tokens = x.flatten(2).transpose(1, 2) + self.pos_embed
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        return tokens.mean(dim=1)       # (B, dim)


# =========================================================================
# Generic wrapper: backbone + feature MLP + fusion + prediction head
# =========================================================================

class GenericSrgNet(nn.Module):
    """Generic Srg model with a backbone, feature MLP, and selectable fusion module."""

    def __init__(
        self,
        backbone: nn.Module,
        n_features: int = 18,
        feat_hidden: int = 64,
        fusion: str = "cross_attn",
        head_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cnn = backbone
        v_dim = backbone.out_channels
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_features, feat_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_hidden, feat_hidden),
            nn.ReLU(inplace=True),
        )
        fusion_cls = FUSION_REGISTRY[fusion]
        self.fusion = fusion_cls(voxel_dim=v_dim, feat_dim=feat_hidden, out_dim=head_hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_hidden, 1))

    def forward(self, voxel: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        h_v = self.cnn(voxel)
        h_f = self.feat_mlp(features)
        h = self.fusion(h_v, h_f)
        return self.head(h).squeeze(-1)   # linear output; train.py handles range recovery/clipping


# =========================================================================
# Tau-guided channel gating modules
# =========================================================================

class TauGate(nn.Module):
    """Tau-guided attention module for modulating the voxel pathway.

    The scalar tau feature is passed through a small MLP and used as a sigmoid gate.
    """

    def __init__(self, voxel_dim: int, tau_idx: int = 1, hidden: int = 16) -> None:
        super().__init__()
        self.tau_idx = tau_idx
        self.gate_net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, voxel_dim),
            nn.Sigmoid(),
        )

    def forward(self, h_v: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        tau = features[:, self.tau_idx:self.tau_idx + 1]
        gate = self.gate_net(tau)
        return h_v * gate


class ResidualTauGate(nn.Module):
    """Residual tau-guided gate for bidirectional modulation of voxel features."""

    def __init__(
        self,
        voxel_dim: int,
        tau_idx: int = 1,
        hidden: int = 16,
        gamma_init: float = 0.1,
        max_gamma: float = 0.5,
    ) -> None:
        super().__init__()
        if not (0.0 < max_gamma <= 1.0):
            raise ValueError("max_gamma must satisfy 0 < max_gamma <= 1")
        if not (0.0 < gamma_init < max_gamma):
            raise ValueError("gamma_init must satisfy 0 < gamma_init < max_gamma")

        self.tau_idx = tau_idx
        self.max_gamma = float(max_gamma)

        init_ratio = gamma_init / max_gamma
        raw_gamma_init = math.log(init_ratio / (1.0 - init_ratio))
        self.raw_gamma = nn.Parameter(torch.tensor(raw_gamma_init, dtype=torch.float32))

        self.gate_net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, voxel_dim),
            nn.Tanh(),
        )

    @property
    def gamma(self) -> torch.Tensor:
        return self.max_gamma * torch.sigmoid(self.raw_gamma)

    def forward(self, h_v: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        tau = features[:, self.tau_idx:self.tau_idx + 1]
        gate = self.gate_net(tau)
        scale = 1.0 + self.gamma * gate
        return h_v * scale


# =========================================================================
# PoreFlowNet: original TauGate-based surrogate model
# =========================================================================

class PoreFlowNet(nn.Module):
    """PoreFlowNet surrogate for residual gas saturation prediction."""

    def __init__(self, n_features: int = 18, tau_idx: int = 1) -> None:
        super().__init__()
        # CNN backbone (Simple 76K)
        from model import Conv3dBlock
        self.cnn = nn.Sequential(
            Conv3dBlock(1, 16),
            Conv3dBlock(16, 32),
            Conv3dBlock(32, 64),
        )
        self.gap = nn.AdaptiveAvgPool3d(1)
        v_dim = 64

        # Tau-guided channel gating.
        self.tau_gate = TauGate(v_dim, tau_idx=tau_idx)

        # features MLP
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        f_dim = 64

        # Cross-attention fusion (q=features, k=v=voxel).
        embed_dim = max(v_dim, f_dim)
        self.q_proj = nn.Linear(f_dim, embed_dim)
        self.k_proj = nn.Linear(v_dim, embed_dim)
        self.v_proj = nn.Linear(v_dim, embed_dim)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.f_residual = nn.Linear(f_dim, embed_dim)

        # head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, voxel: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # voxel pathway
        h_v = self.gap(self.cnn(voxel)).flatten(1)         # (B, 64)
        h_v = self.tau_gate(h_v, features)                  # tau-guided gate

        # feature pathway
        h_f = self.feat_mlp(features)                       # (B, 64)

        # Single-token cross-attention.
        q = self.q_proj(h_f); k = self.k_proj(h_v); v = self.v_proj(h_v)
        scale = q.shape[-1] ** -0.5
        attn = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) * scale)
        attended = self.attn_norm(attn * v + self.f_residual(h_f))

        return self.head(attended).squeeze(-1)              # linear output, clipped during evaluation


# === PoreFlowNet ablation variants ===

class PoreFlowNet_NoTauGate(PoreFlowNet):
    def forward(self, voxel, features):
        h_v = self.gap(self.cnn(voxel)).flatten(1)
        h_f = self.feat_mlp(features)
        q = self.q_proj(h_f); k = self.k_proj(h_v); v = self.v_proj(h_v)
        scale = q.shape[-1] ** -0.5
        attn = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) * scale)
        attended = self.attn_norm(attn * v + self.f_residual(h_f))
        return self.head(attended).squeeze(-1)


class PoreFlowNet_NoCrossAttn(nn.Module):
    """PoreFlowNet ablation variant using concat fusion."""

    def __init__(self, n_features: int = 18, tau_idx: int = 1) -> None:
        super().__init__()
        from model import Conv3dBlock
        self.cnn = nn.Sequential(Conv3dBlock(1, 16), Conv3dBlock(16, 32), Conv3dBlock(32, 64))
        self.gap = nn.AdaptiveAvgPool3d(1)
        v_dim = 64
        self.tau_gate = TauGate(v_dim, tau_idx=tau_idx)
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_features, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 64), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(v_dim + 64, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )

    def forward(self, voxel, features):
        h_v = self.gap(self.cnn(voxel)).flatten(1)
        h_v = self.tau_gate(h_v, features)
        h_f = self.feat_mlp(features)
        return self.head(torch.cat([h_v, h_f], dim=1)).squeeze(-1)


# =========================================================================
# PoreDualNet: dual-path TauGate-based surrogate model
# =========================================================================

class PoreDualNet(nn.Module):
    """Dual-path surrogate with ResidualTauGate and a feature shortcut."""

    def __init__(
        self,
        n_features: int = 18,
        tau_idx: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()

        cnn_channels = kwargs.pop("cnn_channels", (16, 32, 64))
        feat_hidden = kwargs.pop("feat_hidden", 32)
        head_hidden = kwargs.pop("head_hidden", 32)
        dropout = kwargs.pop("dropout", 0.1)
        shortcut_init = kwargs.pop("shortcut_init", 0.1)
        shortcut_max = kwargs.pop("shortcut_max", 1.0)

        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(
                f"PoreDualNet.__init__() "
                f"got an unexpected keyword argument '{unexpected}'"
            )

        # ---- CNN Backbone ----
        from model import Conv3dBlock
        chs = (1,) + tuple(cnn_channels)
        self.cnn = nn.Sequential(
            *[Conv3dBlock(chs[i], chs[i + 1]) for i in range(len(cnn_channels))]
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.cnn_out_dim = cnn_channels[-1]

        # ---- Feature MLP ----
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_features, feat_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_hidden, feat_hidden),
            nn.ReLU(inplace=True),
        )

        # ---- Prediction Head (delta correction path) ----
        self.head = nn.Sequential(
            nn.Linear(self.cnn_out_dim + feat_hidden, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        # ---- Residual Tau Gate ----
        self.tau_gate = ResidualTauGate(self.cnn_out_dim, tau_idx=tau_idx)

        # ---- Feature Shortcut (baseline path) ----
        self.feat_shortcut = nn.Linear(n_features, 1)
        nn.init.zeros_(self.feat_shortcut.weight)
        nn.init.zeros_(self.feat_shortcut.bias)

        # ---- Learnable shortcut_scale ----
        if not (0.0 < shortcut_max <= 1.0):
            raise ValueError("shortcut_max must satisfy 0 < shortcut_max <= 1")
        if not (0.0 < shortcut_init < shortcut_max):
            raise ValueError(
                "shortcut_init must satisfy 0 < shortcut_init < shortcut_max"
            )

        self.shortcut_max = float(shortcut_max)
        init_ratio = shortcut_init / shortcut_max
        raw_shortcut_init = math.log(init_ratio / (1.0 - init_ratio))
        self.raw_shortcut_scale = nn.Parameter(
            torch.tensor(raw_shortcut_init, dtype=torch.float32)
        )

    @property
    def shortcut_scale(self) -> torch.Tensor:
        return self.shortcut_max * torch.sigmoid(self.raw_shortcut_scale)

    def forward(self, voxel: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        h_v = self.cnn(voxel)
        h_v = self.global_pool(h_v).flatten(1)
        h_v = self.tau_gate(h_v, features)                # residual tau gating

        h_f = self.feat_mlp(features)

        h = torch.cat([h_v, h_f], dim=1)
        main_logit = self.head(h).squeeze(-1)              # CNN correction path

        shortcut_logit = self.feat_shortcut(features).squeeze(-1)  # feature baseline path

        return torch.sigmoid(main_logit + self.shortcut_scale * shortcut_logit)


# --- PoreDualNet ablation variants ---

class PoreDualNet_NoTauGate(PoreDualNet):
    """Ablation variant without ResidualTauGate."""

    def __init__(self, n_features: int = 18, tau_idx: int = 1, **kwargs) -> None:
        super().__init__(n_features=n_features, tau_idx=tau_idx, **kwargs)
        # Replace tau_gate with an identity mapping.
        self.tau_gate = nn.Identity()

    def forward(self, voxel, features):
        h_v = self.global_pool(self.cnn(voxel)).flatten(1)
        h_v = self.tau_gate(h_v)
        h_f = self.feat_mlp(features)
        h = torch.cat([h_v, h_f], dim=1)
        main_logit = self.head(h).squeeze(-1)
        shortcut_logit = self.feat_shortcut(features).squeeze(-1)
        return torch.sigmoid(main_logit + self.shortcut_scale * shortcut_logit)


class PoreDualNet_NoShortcut(PoreDualNet):
    """Ablation variant without the feature-shortcut path."""

    def forward(self, voxel, features):
        h_v = self.global_pool(self.cnn(voxel)).flatten(1)
        h_v = self.tau_gate(h_v, features)
        h_f = self.feat_mlp(features)
        h = torch.cat([h_v, h_f], dim=1)
        main_logit = self.head(h).squeeze(-1)
        return torch.sigmoid(main_logit)


class VoxelOnlyCNN(nn.Module):
    """Voxel-only baseline without descriptor features."""

    def __init__(self, n_features: int = 18) -> None:
        super().__init__()
        from model import Conv3dBlock
        self.cnn = nn.Sequential(Conv3dBlock(1, 16), Conv3dBlock(16, 32), Conv3dBlock(32, 64))
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, voxel, features):
        h = self.gap(self.cnn(voxel)).flatten(1)
        return self.head(h).squeeze(-1)


def make_model(name: str, n_features: int = 18) -> nn.Module:
    """Factory for selecting a model by CLI name."""
    # Lightweight baseline models defined in model.py.
    if name == "simple_taugate":
        from model import SimpleTauGateNet
        return SimpleTauGateNet(n_features=n_features, tau_idx=1)
    if name == "simple_sigmoid":
        from model import SimpleSrgNetSigmoid
        return SimpleSrgNetSigmoid(n_features=n_features)
    if name == "simple":
        from model import SimpleSrgNet
        return SimpleSrgNet(n_features=n_features)
    if name == "poreflownet":
        return PoreFlowNet(n_features=n_features, tau_idx=1)
    if name == "poreflownet_no_taugate":
        return PoreFlowNet_NoTauGate(n_features=n_features, tau_idx=1)
    if name == "poreflownet_no_crossattn":
        return PoreFlowNet_NoCrossAttn(n_features=n_features, tau_idx=1)
    if name == "poredualnet":
        return PoreDualNet(n_features=n_features, tau_idx=1)
    if name == "poredualnet_no_taugate":
        return PoreDualNet_NoTauGate(n_features=n_features, tau_idx=1)
    if name == "poredualnet_no_shortcut":
        return PoreDualNet_NoShortcut(n_features=n_features, tau_idx=1)
    if name == "voxel_only_cnn":
        return VoxelOnlyCNN(n_features=n_features)
    if name == "resnet10_tiny_crossattn":
        return GenericSrgNet(ResNet10Tiny(), n_features=n_features, fusion="cross_attn")
    if name == "ms_porenet_crossattn":
        return GenericSrgNet(MSPoreNet(), n_features=n_features, fusion="cross_attn")
    if name == "porecoat_crossattn":
        return GenericSrgNet(PoreCoAt(), n_features=n_features, fusion="cross_attn")
    if name == "poreformer_crossattn":
        return GenericSrgNet(PoreFormer(), n_features=n_features, fusion="cross_attn")
    raise ValueError(name)


class ResNetSrgNet(nn.Module):
    """3D ResNet + feature MLP + selectable fusion + prediction head.

    Args:
        n_features: Dimensionality of numerical descriptor features.
        base_width: Initial channel width of the ResNet backbone.
        layers: Number of blocks in each of the four stages.
        feat_hidden: Hidden dimension of the feature MLP.
        fusion: 'concat' / 'cross_attn' / 'film'
        head_hidden: Hidden dimension after fusion.
        dropout: Global dropout rate.
    """

    def __init__(
        self,
        n_features: int = 18,
        base_width: int = 32,
        layers: tuple[int, ...] = (2, 2, 2, 2),
        feat_hidden: int = 64,
        fusion: str = "concat",
        head_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cnn = ResNet3D(in_channels=1, base_width=base_width, layers=layers)
        v_dim = self.cnn.out_channels

        self.feat_mlp = nn.Sequential(
            nn.Linear(n_features, feat_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_hidden, feat_hidden),
            nn.ReLU(inplace=True),
        )

        fusion_cls = FUSION_REGISTRY[fusion]
        self.fusion = fusion_cls(voxel_dim=v_dim, feat_dim=feat_hidden, out_dim=head_hidden)
        self.fusion_name = fusion

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, voxel: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        h_v = self.cnn(voxel)
        h_f = self.feat_mlp(features)
        h = self.fusion(h_v, h_f)
        return self.head(h).squeeze(-1)   # linear output; train.py handles range recovery/clipping
