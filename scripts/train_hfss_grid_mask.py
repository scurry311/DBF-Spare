"""Train a grid-aware aperture model for the HFSS multitask dataset.

The model treats the 256-element URA as a 16x16 aperture instead of a flat
vector. It combines per-element geometry, target steering phase maps, and
global task conditioning, then predicts 16x16 active-element logits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_hfss_steering_mask import (
    analytic_weights_from_gate,
    gate_from_logits,
    physics_metrics_loss,
    resolve_device,
    steering_weights,
    summarize_group,
)
from train_hfss_surrogate import (
    DEFAULT_DATASET_DIR,
    KMAX,
    K_VALUES,
    NUM_ELEMENTS,
    EvalSummary,
    HfssArrayDataset,
    build_features,
    compute_af_metrics,
    hard_gate_from_logits,
    load_split_manifest,
    make_grid,
    safe_mean,
    target_dirs_torch,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--condition-dim", type=int, default=32)
    parser.add_argument("--attention-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--psll-margin-db", type=float, default=2.5)
    parser.add_argument("--spread-margin-db", type=float, default=5.0)
    parser.add_argument("--binarize-weight", type=float, default=0.02)
    parser.add_argument("--mask-bce-weight", type=float, default=0.16)
    parser.add_argument("--weak-peak-weight", type=float, default=0.04)
    parser.add_argument("--logit-l2-weight", type=float, default=1.0e-4)
    parser.add_argument("--k46-sample-weight", type=float, default=2.5)
    parser.add_argument("--train-eval-every", type=int, default=20)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class GridMaskNet(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        element_ixiy: np.ndarray,
        positions_lambda: np.ndarray,
        channels: int = 96,
        condition_dim: int = 32,
        attention_layers: int = 1,
    ):
        super().__init__()
        ixiy = element_ixiy.astype(np.int64)
        self.nx = int(ixiy[:, 0].max()) + 1
        self.ny = int(ixiy[:, 1].max()) + 1
        self.channels = int(channels)
        self.condition_dim = int(condition_dim)
        self.attention_layers = int(attention_layers)

        static = self._build_static_maps(ixiy)
        self.register_buffer("static_maps", torch.from_numpy(static), persistent=False)
        self.register_buffer("positions", torch.from_numpy(positions_lambda.astype(np.float32)), persistent=False)
        self.register_buffer("element_ix", torch.from_numpy(ixiy[:, 0]), persistent=False)
        self.register_buffer("element_iy", torch.from_numpy(ixiy[:, 1]), persistent=False)

        dynamic_channels = KMAX * 2 + 1
        in_channels = static.shape[0] + dynamic_channels + condition_dim
        self.condition = nn.Sequential(
            nn.Linear(feature_dim, channels),
            nn.LayerNorm(channels),
            nn.SiLU(),
            nn.Linear(channels, condition_dim),
            nn.SiLU(),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
        )
        self.conv_blocks = nn.Sequential(
            ResidualConvBlock(channels),
            ResidualConvBlock(channels),
            ResidualConvBlock(channels),
        )
        if attention_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=4,
                dim_feedforward=channels * 2,
                dropout=0.05,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.attention = nn.TransformerEncoder(encoder_layer, num_layers=attention_layers)
        else:
            self.attention = None
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
        )

    def _build_static_maps(self, ixiy: np.ndarray) -> np.ndarray:
        nx = int(ixiy[:, 0].max()) + 1
        ny = int(ixiy[:, 1].max()) + 1
        ix_grid, iy_grid = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        x = (ix_grid.astype(np.float32) - (nx - 1) / 2.0) / max((nx - 1) / 2.0, 1.0)
        y = (iy_grid.astype(np.float32) - (ny - 1) / 2.0) / max((ny - 1) / 2.0, 1.0)
        radius = np.sqrt(x * x + y * y).astype(np.float32)
        edge = np.minimum.reduce([ix_grid, iy_grid, nx - 1 - ix_grid, ny - 1 - iy_grid]).astype(np.float32)
        edge = edge / max(float(edge.max()), 1.0)
        checker = (((ix_grid + iy_grid) % 2) * 2 - 1).astype(np.float32)
        return np.stack([x, y, radius, edge, checker], axis=0).astype(np.float32)

    def _phase_maps(self, targets_deg: torch.Tensor, task_valid: torch.Tensor) -> torch.Tensor:
        batch = targets_deg.shape[0]
        dirs = target_dirs_torch(targets_deg)
        phase = 2.0 * math.pi * torch.einsum("bkd,nd->bkn", dirs, self.positions)
        valid = task_valid[:, :, None].to(phase.dtype)
        cos_phase = torch.cos(phase) * valid
        sin_phase = torch.sin(phase) * valid
        phase_features = torch.cat([cos_phase, sin_phase], dim=1)

        coherent = torch.sqrt(
            (cos_phase.sum(dim=1) / task_valid.sum(dim=1, keepdim=True).clamp_min(1.0)).square()
            + (sin_phase.sum(dim=1) / task_valid.sum(dim=1, keepdim=True).clamp_min(1.0)).square()
        )
        phase_features = torch.cat([phase_features, coherent[:, None, :]], dim=1)

        maps = torch.zeros(
            batch,
            phase_features.shape[1],
            self.nx,
            self.ny,
            dtype=phase_features.dtype,
            device=phase_features.device,
        )
        maps[:, :, self.element_ix, self.element_iy] = phase_features
        return maps

    def forward(
        self,
        features: torch.Tensor,
        targets_deg: torch.Tensor,
        task_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch = features.shape[0]
        static = self.static_maps[None, :, :, :].expand(batch, -1, -1, -1)
        phase_maps = self._phase_maps(targets_deg, task_valid)
        cond = self.condition(features)[:, :, None, None].expand(-1, -1, self.nx, self.ny)
        x = torch.cat([static, phase_maps, cond], dim=1)
        x = self.conv_blocks(self.stem(x))
        if self.attention is not None:
            tokens = x.flatten(2).transpose(1, 2)
            tokens = self.attention(tokens)
            x = tokens.transpose(1, 2).reshape(batch, self.channels, self.nx, self.ny)
        grid_logits = self.head(x).squeeze(1)
        return grid_logits[:, self.element_ix, self.element_iy]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    positions: torch.Tensor,
    grid_dirs: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    total_n = 0
    bce = nn.BCEWithLogitsLoss()
    for batch in loader:
        features = batch["features"].to(device)
        target_mask = batch["mask"].to(device)
        targets_deg = batch["targets_deg"].to(device)
        task_valid = batch["task_valid"].to(device)
        num_active = batch["num_active"].to(device)

        logits = model(features, targets_deg, task_valid)
        gate = gate_from_logits(logits, num_active, hard=False)
        weights = steering_weights(gate, targets_deg, task_valid, num_active, positions)
        physics_loss, stats = physics_metrics_loss(
            weights,
            targets_deg,
            task_valid,
            positions,
            grid_dirs,
            args.psll_margin_db,
            args.spread_margin_db,
            args.weak_peak_weight,
        )
        bce_loss = bce(logits, target_mask)
        bin_loss = (gate * (1.0 - gate)).mean()
        logit_l2 = logits.square().mean()
        loss = (
            physics_loss
            + args.mask_bce_weight * bce_loss
            + args.binarize_weight * bin_loss
            + args.logit_l2_weight * logit_l2
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        bs = int(features.shape[0])
        total_n += bs
        for key, value in {
            "loss": loss,
            "physics_loss": physics_loss,
            "bce_loss": bce_loss,
            "bin_loss": bin_loss,
            "logit_l2": logit_l2,
            **stats,
        }.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * bs
    return {key: value / max(total_n, 1) for key, value in totals.items()}


@torch.no_grad()
def collect_grid_logits(
    model: nn.Module,
    dataset: HfssArrayDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    idx_chunks: list[np.ndarray] = []
    logit_chunks: list[np.ndarray] = []
    for batch in loader:
        logits = model(
            batch["features"].to(device),
            batch["targets_deg"].to(device),
            batch["task_valid"].to(device),
        )
        idx_chunks.append(batch["index"].numpy())
        logit_chunks.append(logits.cpu().numpy().astype(np.float32))
    idx = np.concatenate(idx_chunks)
    logits = np.concatenate(logit_chunks)
    order = np.argsort(idx)
    return idx[order], logits[order]


def evaluate(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    features: np.ndarray,
    splits: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[list[EvalSummary], dict[str, float]]:
    all_idx = np.arange(arrays["k_values"].shape[0], dtype=np.int64)
    dataset = HfssArrayDataset(arrays, features, all_idx)
    _idx, logits = collect_grid_logits(model, dataset, device, batch_size)
    hard_gate = hard_gate_from_logits(logits, arrays["num_active"].astype(np.float32))
    soft_active = 1.0 / (1.0 + np.exp(-logits))
    soft_active_mean = soft_active.mean(axis=1)
    target_gate = arrays["masks"].astype(np.float32)
    inter = (hard_gate * target_gate).sum(axis=1)
    union = np.maximum(hard_gate + target_gate, 0.0).clip(0.0, 1.0).sum(axis=1)
    mask_iou = inter / np.maximum(union, 1.0)

    model_weights = analytic_weights_from_gate(hard_gate, arrays)
    teacher_weights = arrays["task_weights_real_imag"][..., 0] + 1j * arrays["task_weights_real_imag"][..., 1]
    model_scaled = np.stack([model_weights.real, model_weights.imag], axis=-1) * arrays["num_active"][:, None, None, None]
    teacher_scaled = arrays["task_weights_real_imag"].astype(np.float32) * arrays["num_active"][:, None, None, None]
    valid = arrays["task_valid"][:, None, :, None].astype(np.float32)
    weight_rmse = np.sqrt(
        ((model_scaled - teacher_scaled) ** 2 * valid).sum(axis=(1, 2, 3))
        / np.maximum(valid.sum(axis=(1, 2, 3)) * NUM_ELEMENTS * 2, 1.0)
    )

    _theta, _phi, eval_grid = make_grid(theta_step=2.0, phi_step=5.0)
    teacher_metrics = compute_af_metrics(
        teacher_weights.astype(np.complex64),
        arrays["targets_deg"].astype(np.float32),
        arrays["task_valid"].astype(np.float32),
        arrays["positions_lambda"].astype(np.float32),
        eval_grid,
    )
    model_metrics = compute_af_metrics(
        model_weights.astype(np.complex64),
        arrays["targets_deg"].astype(np.float32),
        arrays["task_valid"].astype(np.float32),
        arrays["positions_lambda"].astype(np.float32),
        eval_grid,
    )
    eval_arrays: dict[str, np.ndarray | dict[str, np.ndarray]] = {
        "teacher": teacher_metrics,
        "model": model_metrics,
        "mask_iou": mask_iou.astype(np.float32),
        "soft_active": soft_active_mean.astype(np.float32),
        "active_ratio": arrays["active_ratios_requested"].astype(np.float32),
        "weight_rmse_scaled": weight_rmse.astype(np.float32),
    }

    rows: list[EvalSummary] = []
    for split, idx in splits.items():
        rows.append(summarize_group(split, "all", "all", idx, eval_arrays))
        for k in K_VALUES:
            k_idx = idx[arrays["k_values"][idx] == k]
            rows.append(summarize_group(split, k, "all", k_idx, eval_arrays))
            for active_ratio in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                sub = k_idx[np.isclose(arrays["active_ratios_requested"][k_idx], active_ratio)]
                rows.append(summarize_group(split, k, f"{active_ratio:.1f}", sub, eval_arrays))

    test_idx = splits["test"]
    k46 = test_idx[np.isin(arrays["k_values"][test_idx], [4, 6])]
    headline = {
        "test_mask_iou_mean": safe_mean(mask_iou[test_idx]),
        "test_soft_active_error_mean": safe_mean(
            np.abs(soft_active_mean[test_idx] - arrays["active_ratios_requested"][test_idx])
        ),
        "test_weight_rmse_scaled": safe_mean(weight_rmse[test_idx]),
        "test_delta_psll_weak_mean_db": safe_mean(
            model_metrics["psll_to_weakest_peak_db"][test_idx]
            - teacher_metrics["psll_to_weakest_peak_db"][test_idx]
        ),
        "test_delta_weak_peak_mean_db": safe_mean(
            model_metrics["target_peak_min_db"][test_idx] - teacher_metrics["target_peak_min_db"][test_idx]
        ),
        "test_k46_delta_psll_weak_mean_db": safe_mean(
            model_metrics["psll_to_weakest_peak_db"][k46] - teacher_metrics["psll_to_weakest_peak_db"][k46]
        ),
        "test_k46_delta_weak_peak_mean_db": safe_mean(
            model_metrics["target_peak_min_db"][k46] - teacher_metrics["target_peak_min_db"][k46]
        ),
    }
    return rows, headline


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset_dir: Path = args.dataset_dir
    if args.out_dir is None:
        out_dir = dataset_dir / "training_runs" / f"grid_mask_{time.strftime('%Y%m%d_%H%M%S')}"
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays_npz = np.load(dataset_dir / "dataset_arrays.npz", allow_pickle=False)
    arrays = {key: arrays_npz[key] for key in arrays_npz.files}
    splits = load_split_manifest(dataset_dir / "training_split_manifest.json")
    features = build_features(
        arrays["k_values"],
        arrays["active_ratios_requested"],
        arrays["targets_deg"],
        arrays["task_valid"],
    )
    train_dataset = HfssArrayDataset(arrays, features, splits["train"])
    if args.k46_sample_weight > 1.0:
        train_idx = splits["train"]
        train_weights = np.ones(train_idx.shape[0], dtype=np.float64)
        train_weights[np.isin(arrays["k_values"][train_idx], [4, 6])] = float(args.k46_sample_weight)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(train_weights),
            num_samples=int(train_idx.shape[0]),
            replacement=True,
        )
        loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = GridMaskNet(
        feature_dim=features.shape[1],
        element_ixiy=arrays["element_ixiy"],
        positions_lambda=arrays["positions_lambda"],
        channels=args.channels,
        condition_dim=args.condition_dim,
        attention_layers=args.attention_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.lr * 0.08,
    )
    positions = torch.from_numpy(arrays["positions_lambda"].astype(np.float32)).to(device)
    _theta, _phi, train_grid_np = make_grid(theta_step=4.0, phi_step=8.0)
    train_grid = torch.from_numpy(train_grid_np).to(device)

    config = {
        "model_type": "grid_mask",
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "device": str(device),
        "feature_dim": int(features.shape[1]),
        "num_train": int(splits["train"].shape[0]),
        "num_val": int(splits["val"].shape[0]),
        "num_test": int(splits["test"].shape[0]),
        "model": {
            "channels": int(args.channels),
            "condition_dim": int(args.condition_dim),
            "attention_layers": int(args.attention_layers),
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Training grid-mask model on {device}; output={out_dir}")
    print(f"splits train/val/test={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    log_rows: list[dict[str, object]] = []
    start = time.time()
    for epoch in range(args.epochs):
        stats = train_one_epoch(model, loader, optimizer, device, positions, train_grid, args)
        scheduler.step()
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **stats}
        log_rows.append(row)
        if epoch == 0 or (epoch + 1) % args.train_eval_every == 0 or epoch + 1 == args.epochs:
            msg = ", ".join(f"{key}={value:.5g}" for key, value in stats.items())
            print(f"epoch {epoch + 1:03d}/{args.epochs}: {msg}")
            write_csv(out_dir / "train_log.csv", log_rows)

    torch.save({"model_state": model.state_dict(), "config": config}, out_dir / "grid_mask_model.pt")
    # Also save the common name so generic evaluators can discover the checkpoint.
    torch.save({"model_state": model.state_dict(), "config": config}, out_dir / "steering_mask_model.pt")
    print("Evaluating grid-mask model...")
    rows, headline = evaluate(model, arrays, features, splits, device, args.batch_size)
    write_csv(out_dir / "eval_summary_by_split_k_active.csv", [asdict(row) for row in rows])
    metrics = {
        "headline": headline,
        "elapsed_s": time.time() - start,
        "outputs": {
            "train_log": str(out_dir / "train_log.csv"),
            "eval_summary": str(out_dir / "eval_summary_by_split_k_active.csv"),
            "model": str(out_dir / "grid_mask_model.pt"),
        },
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
