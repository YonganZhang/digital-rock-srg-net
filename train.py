""" 6-fold leave-one-rock-out CV on 128^3

Lightning module:
    - Input: Voxel grid + tabular features.
    - Output: Srg in (0, 1).
    - Loss: MSE on Srg.
    - Metrics: R2, MAE, RMSE (reported per-fold and aggregated).

CV Protocol:
    6-fold leave-one-rock-out cross-validation. For fold i, the validation set
    comprises all sub-volumes of parent core i; the training set contains the
    remaining 5 cores. Total samples approximately 360; per-fold validation size
    approximately 40-80, training size approximately 280-320.

Usage:
    python train.py --data data/processed/voxel_128.npz --model simple --epochs 30 --gpu 0
    python train.py --data data/processed/voxel_128.npz --model phi    --epochs 30 --gpu 0   # baseline

Note: the dataset .npz is NOT shipped with this repository (raw digital-rock data
belongs to the originating study). See README for the expected cache schema.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from data import CachedNPZ, RockDataset, compute_feat_stats, group_kfold_indices
from model import PhiOnlyBaseline, SimpleSrgNet, SimpleSrgNetSigmoid, SimpleTauGateNet
from models_3d import ResNetSrgNet, make_model

RUNS = Path("runs")
RUNS.mkdir(exist_ok=True)


def r2(yhat: np.ndarray, y: np.ndarray) -> float:
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def metrics(yhat: np.ndarray, y: np.ndarray) -> dict:
    err = yhat - y
    return {
        "n": int(len(y)),
        "R2": r2(yhat, y),
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "y_mean": float(y.mean()),
        "yhat_mean": float(yhat.mean()),
    }


EPS_LOGIT = 1e-3


def to_logit(y: torch.Tensor) -> torch.Tensor:
    yc = y.clamp(EPS_LOGIT, 1.0 - EPS_LOGIT)
    return torch.log(yc / (1.0 - yc))


def from_logit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class SrgLitModule(L.LightningModule):
    """Target transform strategy:

    - 'none': Model outputs raw values directly. Training target is Srg;
      evaluation outputs are clipped to [0, 1].
    - 'logit': Model outputs logits. Training target is logit(Srg);
      evaluation applies sigmoid recovery followed by clipping to [0, 1].
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4,
                 scheduler: str = "none", max_epochs: int = 30,
                 target_transform: str = "logit") -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.wd = weight_decay
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.target_transform = target_transform
        self.val_preds: list[np.ndarray] = []    # Inverse-transformed and clipped to [0, 1]
        self.val_targets: list[np.ndarray] = []  # Original Srg

    def forward(self, voxel, features):
        return self.model(voxel, features)

    def _step(self, batch, stage: str):
        yhat = self(batch["voxel"], batch["features"])
        y_orig = batch["Srg"]
        y_target = to_logit(y_orig) if self.target_transform == "logit" else y_orig
        loss = nn.functional.mse_loss(yhat, y_target)
        self.log(f"{stage}_mse", loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=y_orig.shape[0])
        if stage == "val":
            yhat_np = yhat.detach().cpu().numpy()
            yhat_orig = from_logit(yhat_np) if self.target_transform == "logit" else yhat_np
            yhat_orig = np.clip(yhat_orig, 0.0, 1.0)
            self.val_preds.append(yhat_orig)
            self.val_targets.append(y_orig.detach().cpu().numpy())
        return loss

    def training_step(self, batch, idx):
        return self._step(batch, "train")

    def validation_step(self, batch, idx):
        return self._step(batch, "val")

    def on_validation_epoch_end(self):
        if not self.val_preds:
            return
        yhat = np.concatenate(self.val_preds)
        y = np.concatenate(self.val_targets)
        m = metrics(yhat, y)
        for k, v in m.items():
            if isinstance(v, (int, float)):
                self.log(f"val_{k}", v, prog_bar=(k in {"R2", "MAE"}))
        self.val_preds.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        if self.scheduler == "cosine":
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs, eta_min=self.lr * 0.01)
            return {"optimizer": opt, "lr_scheduler": sch}
        return opt


@dataclass
class FoldResult:
    val_prefix: str
    n_train: int
    n_val: int
    R2: float
    MAE: float
    RMSE: float
    y_mean: float
    yhat_mean: float
    final_train_mse: float
    final_val_mse: float
    pred: list[float]
    target: list[float]
    sample_id: list[str]


def run_one_fold(
    cache: CachedNPZ,
    val_prefix: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    gpu: int,
    augment: bool,
    cli_scheduler: str = "none",
    target_transform: str = "logit",
) -> FoldResult:
    train_idx, val_idx = group_kfold_indices(cache.prefix, val_prefix)
    feat_mean, feat_std = compute_feat_stats(cache.features, train_idx)

    train_ds = RockDataset(cache, train_idx, feat_mean, feat_std, augment=augment)
    val_ds = RockDataset(cache, val_idx, feat_mean, feat_std, augment=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    F = cache.features.shape[1]
    if model_name == "simple":
        model = SimpleSrgNet(n_features=F)
    elif model_name == "simple_sigmoid":
        model = SimpleSrgNetSigmoid(n_features=F)
    elif model_name == "simple_taugate":
        model = SimpleTauGateNet(n_features=F, tau_idx=1)
    elif model_name == "phi":
        model = PhiOnlyBaseline(n_features=F)
    elif model_name == "resnet18_concat":
        model = ResNetSrgNet(n_features=F, fusion="concat")
    elif model_name == "resnet18_crossattn":
        model = ResNetSrgNet(n_features=F, fusion="cross_attn")
    elif model_name == "resnet18_film":
        model = ResNetSrgNet(n_features=F, fusion="film")
    elif model_name in {"resnet10_tiny_crossattn", "ms_porenet_crossattn",
                        "porecoat_crossattn", "poreformer_crossattn",
                        "poreflownet", "poreflownet_no_taugate",
                        "poreflownet_no_crossattn", "voxel_only_cnn",
                        "poredualnet", "poredualnet_no_taugate",
                        "poredualnet_no_shortcut"}:
        model = make_model(model_name, n_features=F)
    else:
        raise ValueError(model_name)

    # PoreDualNet variants include a built-in sigmoid; the 'logit' transform
    # must not be used because it would apply a second sigmoid during evaluation.
    if model_name.startswith("poredualnet"):
        target_transform = "none"

    lit = SrgLitModule(model, scheduler=cli_scheduler, max_epochs=epochs,
                       target_transform=target_transform)
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=[gpu],
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        log_every_n_steps=5,
        callbacks=[EarlyStopping(monitor="val_mse", patience=15, mode="min")],
    )
    trainer.fit(lit, train_dl, val_dl)

    # Re-run validation to obtain clean predictions.
    # Lightning does not guarantee synchronized val lists at the end of fit().
    lit.eval()
    yhat_all, y_all, sid_all = [], [], []
    with torch.no_grad():
        for batch in DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0):
            batch_gpu = {k: (v.to(lit.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            raw_out = lit(batch_gpu["voxel"], batch_gpu["features"]).cpu().numpy()
            # Inverse transform + clip, consistent with _step.
            yhat_orig = from_logit(raw_out) if target_transform == "logit" else raw_out
            yhat_orig = np.clip(yhat_orig, 0.0, 1.0)
            yhat_all.append(yhat_orig)
            y_all.append(batch_gpu["Srg"].cpu().numpy())
            sid_all.extend(batch["sample_id"])
    yhat = np.concatenate(yhat_all)
    y = np.concatenate(y_all)

    m = metrics(yhat, y)
    final_train = float(trainer.callback_metrics.get("train_mse", torch.tensor(float("nan"))).item())
    final_val = float(trainer.callback_metrics.get("val_mse", torch.tensor(float("nan"))).item())

    return FoldResult(
        val_prefix=val_prefix,
        n_train=int(len(train_idx)),
        n_val=int(len(val_idx)),
        R2=m["R2"],
        MAE=m["MAE"],
        RMSE=m["RMSE"],
        y_mean=m["y_mean"],
        yhat_mean=m["yhat_mean"],
        final_train_mse=final_train,
        final_val_mse=final_val,
        pred=yhat.tolist(),
        target=y.tolist(),
        sample_id=list(sid_all),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/processed/voxel_128.npz",
                   help="path to the cached .npz (see README for schema; not shipped)")
    p.add_argument("--model", choices=[
        "simple", "simple_sigmoid", "simple_taugate",
        "phi", "resnet18_concat", "resnet18_crossattn", "resnet18_film",
        "resnet10_tiny_crossattn", "ms_porenet_crossattn",
        "porecoat_crossattn", "poreformer_crossattn",
        "poreflownet", "poreflownet_no_taugate",
        "poreflownet_no_crossattn", "voxel_only_cnn",
        "poredualnet", "poredualnet_no_taugate",
        "poredualnet_no_shortcut",
    ], default="simple")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    p.add_argument("--target-transform", choices=["none", "logit"], default="logit")
    p.add_argument("--tag", type=str, default="default")
    args = p.parse_args()

    cache = CachedNPZ.load(args.data)
    print(f"loaded {cache.n} samples, D={cache.D}, F={cache.features.shape[1]}")
    print(f"prefix dist: {dict(zip(*np.unique(cache.prefix, return_counts=True)))}")

    effective_target_transform = "none" if args.model.startswith("poredualnet") else args.target_transform

    folds: list[FoldResult] = []
    for prefix in sorted(np.unique(cache.prefix)):
        print(f"\n=== fold val={prefix} ===")
        fr = run_one_fold(
            cache, val_prefix=str(prefix),
            model_name=args.model, epochs=args.epochs,
            batch_size=args.batch_size, gpu=args.gpu, augment=args.augment,
            cli_scheduler=args.scheduler,
            target_transform=effective_target_transform,
        )
        print(f"  n_train={fr.n_train} n_val={fr.n_val} R2={fr.R2:.3f} MAE={fr.MAE:.3f} RMSE={fr.RMSE:.3f}")
        folds.append(fr)

    summary = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "augment": args.augment,
        "scheduler": args.scheduler,
        "target_transform": effective_target_transform,
        "requested_target_transform": args.target_transform,
        "n_folds": len(folds),
        "R2_mean": float(np.mean([f.R2 for f in folds])),
        "R2_per_fold": {f.val_prefix: f.R2 for f in folds},
        "MAE_mean": float(np.mean([f.MAE for f in folds])),
        "MAE_per_fold": {f.val_prefix: f.MAE for f in folds},
        "RMSE_mean": float(np.mean([f.RMSE for f in folds])),
        "fold_details": [asdict(f) for f in folds],
    }
    out = RUNS / f"p1_{args.model}_{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n=== SUMMARY ({args.model}) ===")
    print(f"  R2    mean = {summary['R2_mean']:.3f}    per-fold = {summary['R2_per_fold']}")
    print(f"  MAE   mean = {summary['MAE_mean']:.3f}")
    print(f"  RMSE  mean = {summary['RMSE_mean']:.3f}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
