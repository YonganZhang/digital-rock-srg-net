"""Digital rock dataset: Load voxel_{128}.npz and split by parent core prefix."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CachedNPZ:
    voxel: np.ndarray       # (N, D, D, D) uint8
    features: np.ndarray    # (N, F) float32
    Srg: np.ndarray         # (N,) float32 in [0, 1]
    sample_id: np.ndarray   # (N,) <U8
    prefix: np.ndarray      # (N,) <U2  rock id
    feature_names: np.ndarray
    D: int
    n: int

    @classmethod
    def load(cls, path: str | Path) -> "CachedNPZ":
        z = np.load(path, allow_pickle=True)  # sample_id is an object array; allow_pickle=True is required.
        voxel = z["voxel"]
        return cls(
            voxel=voxel,
            features=z["features"].astype(np.float32),
            Srg=z["Srg"].astype(np.float32),
            sample_id=z["sample_id"],
            prefix=z["prefix"],
            feature_names=z["feature_names"],
            D=int(voxel.shape[1]),
            n=int(voxel.shape[0]),
        )


def group_kfold_indices(prefix: np.ndarray, val_prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-rock-out split: val holds one parent core; train uses the rest."""
    val = np.where(prefix == val_prefix)[0]
    train = np.where(prefix != val_prefix)[0]
    return train, val


class RockDataset(Dataset):
    """Single-resolution digital-rock dataset.

    Args:
        cache: CachedNPZ instance for memory-mapped .npz access.
        idx: Sample indices for this subset (derived from group_kfold_indices).
        feat_mean / feat_std: Feature normalization statistics computed on the training set only;
            passed externally to prevent data leakage.
        augment: Whether to apply data augmentation (enabled for training set only).
            Applies 90-degree rotations and flips in the XY plane; no rotation along the Z-axis
            (the through-flow direction).
    """

    def __init__(
        self,
        cache: CachedNPZ,
        idx: np.ndarray,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        augment: bool = False,
    ) -> None:
        self.cache = cache
        self.idx = idx
        self.feat_mean = feat_mean.astype(np.float32)
        self.feat_std = np.where(feat_std > 1e-8, feat_std, 1.0).astype(np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.idx)

    def _augment_xy(self, vox: np.ndarray) -> np.ndarray:
        # vox: (D, D, D) with Z as axis 0; apply 4 rotations and 2 flips in the (Y, X) plane (axes 1, 2).
        k = np.random.randint(4)
        if k:
            vox = np.rot90(vox, k=k, axes=(1, 2))
        if np.random.rand() < 0.5:
            vox = vox[:, ::-1, :]
        if np.random.rand() < 0.5:
            vox = vox[:, :, ::-1]
        return np.ascontiguousarray(vox)

    def __getitem__(self, i: int) -> dict:
        j = int(self.idx[i])
        vox = self.cache.voxel[j]                 # uint8 (D, D, D)
        if self.augment:
            vox = self._augment_xy(vox)
        vox_t = torch.from_numpy(vox.astype(np.float32))[None]   # (1, D, D, D)
        feat = self.cache.features[j]
        feat_norm = (feat - self.feat_mean) / self.feat_std
        return {
            "voxel": vox_t,
            "features": torch.from_numpy(feat_norm),
            "Srg": torch.tensor(self.cache.Srg[j], dtype=torch.float32),
            "sample_id": str(self.cache.sample_id[j]),
            "prefix": str(self.cache.prefix[j]),
        }


def compute_feat_stats(features: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean/std from the train_idx subset only to prevent data leakage."""
    sub = features[train_idx]
    return sub.mean(axis=0), sub.std(axis=0)
