"""Quick test for PoreDualNet.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import CachedNPZ, RockDataset, compute_feat_stats, group_kfold_indices
from models_3d import make_model


def make_synthetic_npz(path: Path, n_rocks: int = 6, samples_per_rock: int = 2) -> None:
    rng = np.random.default_rng(42)

    n = n_rocks * samples_per_rock
    d = 32
    n_features = 18

    voxel = (rng.random((n, d, d, d)) > 0.72).astype(np.uint8)
    features = rng.normal(size=(n, n_features)).astype(np.float32)

    pore_fraction = voxel.mean(axis=(1, 2, 3)).astype(np.float32)
    features[:, 0] = (pore_fraction - pore_fraction.mean()) / (
        pore_fraction.std() + 1e-6
    )

    logits = 1.2 * features[:, 0] + 0.7 * features[:, 1]
    srg = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    prefix = np.repeat([f"rock_{i}" for i in range(n_rocks)], samples_per_rock)
    sample_id = np.array([f"sample_{i:03d}" for i in range(n)])
    feature_names = np.array([f"feature_{i}" for i in range(n_features)])

    # K and logK are included for compatibility with older data loaders.
    k = np.exp(rng.normal(size=n)).astype(np.float32)
    logk = np.log10(k).astype(np.float32)

    np.savez(
        path,
        voxel=voxel,
        features=features,
        Srg=srg,
        K=k,
        logK=logk,
        prefix=prefix,
        sample_id=sample_id,
        feature_names=feature_names,
    )


def build_poredualnet(n_features: int = 18) -> torch.nn.Module:
    try:
        return make_model("poredualnet", n_features=n_features)
    except TypeError:
        return make_model("poredualnet", n_features)


def main() -> None:
    torch.manual_seed(42)

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "synthetic_srg_quicktest.npz"
        make_synthetic_npz(npz_path)

        cache = CachedNPZ.load(npz_path)

        val_prefix = sorted(np.unique(cache.prefix))[0]
        train_idx, val_idx = group_kfold_indices(cache.prefix, val_prefix)
        feat_mean, feat_std = compute_feat_stats(cache.features, train_idx)

        train_ds = RockDataset(
            cache=cache,
            idx=train_idx,
            feat_mean=feat_mean,
            feat_std=feat_std,
            augment=False,
        )

        loader = DataLoader(train_ds, batch_size=2, shuffle=True)

        model = build_poredualnet(n_features=cache.features.shape[1])
        model.train()

        batch = next(iter(loader))
        voxel = batch["voxel"]
        features = batch["features"]
        target = batch["Srg"]

        pred = model(voxel, features)

        assert pred.shape == target.shape, f"Wrong output shape: {pred.shape}"
        assert torch.isfinite(pred).all(), "Prediction contains NaN or Inf"
        assert pred.min() >= 0.0 and pred.max() <= 1.0, "Prediction is outside [0, 1]"

        loss = F.mse_loss(pred, target)
        loss.backward()

        print("Quick test passed.")
        print(f"Synthetic samples: {cache.n}")
        print(f"Voxel size: {cache.D}^3")
        print(f"Feature dimension: {cache.features.shape[1]}")
        print(f"Prediction shape: {tuple(pred.shape)}")
        print(f"Prediction range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
        print(f"Loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()
