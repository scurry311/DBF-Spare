"""Train a quick neural surrogate on the 2400-sample HFSS multitask dataset.

The script is intentionally self-contained so the HFSS quick-model workflow can
be validated without mutating the existing beam-multitask Stage4 artifacts.
It learns:

    K + target directions + active_ratio -> active mask + per-task complex weights

and evaluates whether the learned model improves/harms array-factor PSLL,
weak-target peak, active-ratio control, and split generalization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "hfss_outputs" / "multitask_dataset"
K_VALUES = (1, 2, 4, 6)
KMAX = 6
NUM_ELEMENTS = 256
EPS = 1.0e-12


@dataclass
class EvalSummary:
    split: str
    k: int | str
    active_ratio: str
    n: int
    mask_iou_mean: float
    soft_active_error_mean: float
    weight_rmse_scaled: float
    teacher_target_peak_min_mean_db: float
    model_target_peak_min_mean_db: float
    delta_weak_peak_mean_db: float
    teacher_psll_weak_mean_db: float
    model_psll_weak_mean_db: float
    delta_psll_weak_mean_db: float
    teacher_psll_weak_p95_db: float
    model_psll_weak_p95_db: float
    teacher_isolation_worst_p05_db: float
    model_isolation_worst_p05_db: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--physics-start-epoch", type=int, default=15)
    parser.add_argument("--physics-weight", type=float, default=0.035)
    parser.add_argument("--psll-margin-db", type=float, default=2.0)
    parser.add_argument("--spread-margin-db", type=float, default=5.0)
    parser.add_argument("--train-eval-every", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unit_vectors(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    return np.stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        axis=-1,
    ).astype(np.float32)


def build_features(
    k_values: np.ndarray,
    active_ratios: np.ndarray,
    targets_deg: np.ndarray,
    task_valid: np.ndarray,
) -> np.ndarray:
    num_samples = k_values.shape[0]
    k_onehot = np.zeros((num_samples, len(K_VALUES)), dtype=np.float32)
    for col, kval in enumerate(K_VALUES):
        k_onehot[:, col] = (k_values == kval).astype(np.float32)
    dirs = unit_vectors(targets_deg[:, :, 0], targets_deg[:, :, 1])
    task_valid_f = task_valid.astype(np.float32)
    dirs = np.where(task_valid_f[:, :, None] > 0.0, dirs, 0.0)
    per_task = np.concatenate([task_valid_f[:, :, None], dirs], axis=-1).reshape(num_samples, -1)
    return np.concatenate(
        [
            k_onehot,
            (k_values.astype(np.float32) / float(KMAX))[:, None],
            active_ratios.astype(np.float32)[:, None],
            per_task.astype(np.float32),
        ],
        axis=1,
    )


def load_split_manifest(path: Path) -> dict[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "train": np.asarray(data["splits"]["train"], dtype=np.int64),
        "val": np.asarray(data["splits"]["val"], dtype=np.int64),
        "test": np.asarray(data["splits"]["test"], dtype=np.int64),
    }


class HfssArrayDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], features: np.ndarray, indices: np.ndarray):
        self.arrays = arrays
        self.features = features.astype(np.float32)
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        k = int(self.arrays["k_values"][idx])
        task_valid = self.arrays["task_valid"][idx].astype(np.float32)
        num_active = float(self.arrays["num_active"][idx])
        # Scale by active count so active element weights have O(1) magnitude.
        weights_scaled = self.arrays["task_weights_real_imag"][idx].astype(np.float32) * num_active
        return {
            "index": torch.tensor(idx, dtype=torch.long),
            "features": torch.from_numpy(self.features[idx]),
            "k": torch.tensor(k, dtype=torch.long),
            "active_ratio": torch.tensor(float(self.arrays["active_ratios_requested"][idx]), dtype=torch.float32),
            "num_active": torch.tensor(num_active, dtype=torch.float32),
            "targets_deg": torch.from_numpy(self.arrays["targets_deg"][idx].astype(np.float32)),
            "task_valid": torch.from_numpy(task_valid),
            "mask": torch.from_numpy(self.arrays["masks"][idx].astype(np.float32)),
            "weights_scaled": torch.from_numpy(weights_scaled),
        }


class SurrogateNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mask_head = nn.Linear(hidden, NUM_ELEMENTS)
        self.weight_head = nn.Linear(hidden, NUM_ELEMENTS * KMAX * 2)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(features)
        mask_logits = self.mask_head(h)
        weights_scaled = torch.tanh(self.weight_head(h)).view(-1, NUM_ELEMENTS, KMAX, 2)
        return mask_logits, weights_scaled


def make_grid(theta_step: float = 3.0, phi_step: float = 6.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_values = np.arange(0.0, 90.0 + 0.1, theta_step, dtype=np.float32)
    phi_values = np.arange(0.0, 360.0, phi_step, dtype=np.float32)
    theta, phi = np.meshgrid(theta_values, phi_values, indexing="ij")
    theta_flat = theta.reshape(-1)
    phi_flat = phi.reshape(-1)
    dirs = unit_vectors(theta_flat, phi_flat)
    return theta_flat, phi_flat, dirs


def steering_from_dirs(positions: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    # Beamlearn convention for evaluating AF: exp(-j 2pi r.u).
    phase = 2.0 * math.pi * dirs @ positions.T
    return torch.exp(-1j * phase.to(torch.complex64))


def outputs_to_weights(
    mask_logits: torch.Tensor,
    weights_scaled: torch.Tensor,
    num_active: torch.Tensor,
    task_valid: torch.Tensor,
    *,
    hard: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = mask_logits.shape[0]
    if hard:
        gate = torch.zeros_like(mask_logits)
        for i in range(batch):
            k_active = int(round(float(num_active[i].detach().cpu().item())))
            top_idx = torch.topk(mask_logits[i], k=k_active, largest=True).indices
            gate[i, top_idx] = 1.0
    else:
        gate = torch.sigmoid(mask_logits)
        target_count = num_active[:, None].clamp_min(1.0)
        gate = gate * (target_count / gate.sum(dim=1, keepdim=True).clamp_min(1.0))
        gate = gate.clamp(0.0, 1.0)
    valid = task_valid[:, None, :, None]
    weights_complex = torch.complex(weights_scaled[..., 0], weights_scaled[..., 1])
    weights_complex = weights_complex * gate[:, :, None] * task_valid[:, None, :]
    weights_complex = weights_complex / num_active[:, None, None].clamp_min(1.0)
    return gate, weights_complex


def target_dirs_torch(targets_deg: torch.Tensor) -> torch.Tensor:
    targets_deg = torch.nan_to_num(targets_deg, nan=0.0)
    theta = torch.deg2rad(targets_deg[..., 0])
    phi = torch.deg2rad(targets_deg[..., 1])
    return torch.stack(
        [
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ],
        dim=-1,
    )


def physics_loss(
    weights_complex: torch.Tensor,
    targets_deg: torch.Tensor,
    task_valid: torch.Tensor,
    positions: torch.Tensor,
    grid_dirs: torch.Tensor,
    psll_margin_db: float,
    spread_margin_db: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch = weights_complex.shape[0]
    combined = weights_complex.sum(dim=2)
    target_dirs = target_dirs_torch(targets_deg)
    target_steering = torch.exp(
        -1j * (2.0 * math.pi * torch.einsum("bkd,nd->bkn", target_dirs, positions)).to(torch.complex64)
    )
    target_response = torch.einsum("bkn,bn->bk", target_steering, combined)
    target_db = 10.0 * torch.log10(torch.abs(target_response).square().clamp_min(EPS))
    valid_bool = task_valid.bool()
    target_db_masked_min = target_db.masked_fill(~valid_bool, 1.0e6).min(dim=1).values
    target_db_masked_max = target_db.masked_fill(~valid_bool, -1.0e6).max(dim=1).values

    grid_steering = steering_from_dirs(positions, grid_dirs)
    grid_response = combined @ grid_steering.T
    grid_db = 10.0 * torch.log10(torch.abs(grid_response).square().clamp_min(EPS))

    dots = torch.einsum("gd,bkd->bgk", grid_dirs, target_dirs).clamp(-1.0, 1.0)
    dists = torch.rad2deg(torch.acos(dots))
    dists = torch.where(valid_bool[:, None, :], dists, torch.full_like(dists, 999.0))
    side_mask = dists.min(dim=2).values > 8.0
    side_db = grid_db.masked_fill(~side_mask, -1.0e6).max(dim=1).values

    psll_weak = side_db - target_db_masked_min
    spread = target_db_masked_max - target_db_masked_min
    psll_penalty = torch.relu(psll_weak + psll_margin_db).square().mean()
    spread_penalty = torch.relu(spread - spread_margin_db).square().mean()
    weak_peak_reward = -target_db_masked_min.mean() * 0.01
    loss = psll_penalty + 0.25 * spread_penalty + weak_peak_reward
    return loss, {
        "physics_psll_weak_db": psll_weak.detach().mean(),
        "physics_spread_db": spread.detach().mean(),
        "physics_weak_peak_db": target_db_masked_min.detach().mean(),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    positions: torch.Tensor,
    grid_dirs: torch.Tensor,
    epoch: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = defaultdict_float()
    count = 0
    bce = nn.BCEWithLogitsLoss()
    for batch in loader:
        features = batch["features"].to(device)
        target_mask = batch["mask"].to(device)
        target_weights = batch["weights_scaled"].to(device)
        task_valid = batch["task_valid"].to(device)
        num_active = batch["num_active"].to(device)
        active_ratio = batch["active_ratio"].to(device)
        targets_deg = batch["targets_deg"].to(device)

        mask_logits, weights_scaled = model(features)
        mask_loss = bce(mask_logits, target_mask)
        valid_weight = task_valid[:, None, :, None]
        weight_loss = ((weights_scaled - target_weights) * valid_weight).square().sum() / (
            valid_weight.sum().clamp_min(1.0) * float(NUM_ELEMENTS * 2)
        )
        soft_active = torch.sigmoid(mask_logits).mean(dim=1)
        active_loss = (soft_active - active_ratio).square().mean()
        total_loss = mask_loss + 0.30 * weight_loss + 3.0 * active_loss

        phys_loss_value = torch.zeros((), device=device)
        phys_stats: dict[str, torch.Tensor] = {}
        if epoch >= args.physics_start_epoch and args.physics_weight > 0.0:
            _, soft_weights = outputs_to_weights(mask_logits, weights_scaled, num_active, task_valid, hard=False)
            phys_loss_value, phys_stats = physics_loss(
                soft_weights,
                targets_deg,
                task_valid,
                positions,
                grid_dirs,
                args.psll_margin_db,
                args.spread_margin_db,
            )
            total_loss = total_loss + args.physics_weight * phys_loss_value

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = int(features.shape[0])
        count += bs
        totals["loss"] = totals.get("loss", 0.0) + float(total_loss.detach().cpu()) * bs
        totals["mask_loss"] = totals.get("mask_loss", 0.0) + float(mask_loss.detach().cpu()) * bs
        totals["weight_loss"] = totals.get("weight_loss", 0.0) + float(weight_loss.detach().cpu()) * bs
        totals["active_loss"] = totals.get("active_loss", 0.0) + float(active_loss.detach().cpu()) * bs
        totals["physics_loss"] = totals.get("physics_loss", 0.0) + float(phys_loss_value.detach().cpu()) * bs
        for key, value in phys_stats.items():
            totals[key] = totals.get(key, 0.0) + float(value.cpu()) * bs
    return {key: value / max(count, 1) for key, value in totals.items()}


def defaultdict_float() -> dict[str, float]:
    return {}


def add_metric(totals: dict[str, float], key: str, value: float, weight: int) -> None:
    totals[key] = totals.get(key, 0.0) + value * weight


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    dataset: HfssArrayDataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {
        "index": [],
        "mask_logits": [],
        "weights_scaled": [],
        "soft_active": [],
    }
    for batch in loader:
        features = batch["features"].to(device)
        mask_logits, weights_scaled = model(features)
        chunks["index"].append(batch["index"].numpy())
        chunks["mask_logits"].append(mask_logits.cpu().numpy().astype(np.float32))
        chunks["weights_scaled"].append(weights_scaled.cpu().numpy().astype(np.float32))
        chunks["soft_active"].append(torch.sigmoid(mask_logits).mean(dim=1).cpu().numpy().astype(np.float32))
    return {key: np.concatenate(value, axis=0) for key, value in chunks.items()}


def hard_gate_from_logits(mask_logits: np.ndarray, num_active: np.ndarray) -> np.ndarray:
    gates = np.zeros_like(mask_logits, dtype=np.float32)
    for i in range(mask_logits.shape[0]):
        k = int(round(float(num_active[i])))
        idx = np.argpartition(mask_logits[i], -k)[-k:]
        gates[i, idx] = 1.0
    return gates


def build_eval_grid() -> tuple[np.ndarray, np.ndarray]:
    _, _, dirs = make_grid(theta_step=2.0, phi_step=5.0)
    return dirs, np.arange(dirs.shape[0], dtype=np.int64)


def compute_af_metrics(
    weights: np.ndarray,
    targets_deg: np.ndarray,
    task_valid: np.ndarray,
    positions: np.ndarray,
    grid_dirs: np.ndarray,
) -> dict[str, np.ndarray]:
    num_samples = weights.shape[0]
    combined = weights.sum(axis=2)
    grid_phase = 2.0 * np.pi * (grid_dirs @ positions.T)
    grid_steer = np.exp(-1j * grid_phase).astype(np.complex64)
    grid_unit = grid_dirs.astype(np.float32)
    target_peak_min = np.zeros(num_samples, dtype=np.float32)
    target_peak_mean = np.zeros(num_samples, dtype=np.float32)
    target_spread = np.zeros(num_samples, dtype=np.float32)
    psll_weak = np.zeros(num_samples, dtype=np.float32)
    isolation_worst = np.full(num_samples, np.nan, dtype=np.float32)

    for i in range(num_samples):
        valid = task_valid[i].astype(bool)
        k = int(valid.sum())
        target_dirs_np = unit_vectors(targets_deg[i, :, 0], targets_deg[i, :, 1])
        target_phase = 2.0 * np.pi * (target_dirs_np @ positions.T)
        target_steer = np.exp(-1j * target_phase).astype(np.complex64)
        target_resp = target_steer[valid] @ combined[i]
        target_db = 10.0 * np.log10(np.maximum(np.abs(target_resp) ** 2, EPS))
        target_peak_min[i] = float(np.min(target_db))
        target_peak_mean[i] = float(np.mean(target_db))
        target_spread[i] = float(np.max(target_db) - np.min(target_db))

        grid_resp = grid_steer @ combined[i]
        grid_db = 10.0 * np.log10(np.maximum(np.abs(grid_resp) ** 2, EPS))
        dots = np.clip(grid_unit @ target_dirs_np[valid].T, -1.0, 1.0)
        dists = np.rad2deg(np.arccos(dots))
        side_mask = dists.min(axis=1) > 8.0
        side_max = float(np.max(grid_db[side_mask])) if np.any(side_mask) else float(np.nan)
        psll_weak[i] = side_max - target_peak_min[i]

        if k > 1:
            task_weights = weights[i][:, valid]
            response = target_steer[valid] @ task_weights
            power = np.abs(response) ** 2
            iso_vals = []
            for task_idx in range(k):
                desired = float(power[task_idx, task_idx])
                leakage = max(float(power[other, task_idx]) for other in range(k) if other != task_idx)
                iso_vals.append(10.0 * math.log10((desired + EPS) / (leakage + EPS)))
            isolation_worst[i] = float(np.min(iso_vals))

    return {
        "target_peak_min_db": target_peak_min,
        "target_peak_mean_db": target_peak_mean,
        "target_spread_db": target_spread,
        "psll_to_weakest_peak_db": psll_weak,
        "isolation_worst_db": isolation_worst,
    }


def safe_percentile(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def safe_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(values.mean())


def summarize_group(
    split: str,
    k_label: int | str,
    active_label: str,
    idx: np.ndarray,
    eval_arrays: dict[str, np.ndarray],
) -> EvalSummary:
    teacher = eval_arrays["teacher"]
    model = eval_arrays["model"]
    n = int(idx.size)
    mask_iou = eval_arrays["mask_iou"][idx]
    active_error = np.abs(eval_arrays["soft_active"][idx] - eval_arrays["active_ratio"][idx])
    weight_rmse = eval_arrays["weight_rmse_scaled"][idx]
    teacher_weak = teacher["target_peak_min_db"][idx]
    model_weak = model["target_peak_min_db"][idx]
    teacher_psll = teacher["psll_to_weakest_peak_db"][idx]
    model_psll = model["psll_to_weakest_peak_db"][idx]
    teacher_iso = teacher["isolation_worst_db"][idx]
    model_iso = model["isolation_worst_db"][idx]
    return EvalSummary(
        split=split,
        k=k_label,
        active_ratio=active_label,
        n=n,
        mask_iou_mean=safe_mean(mask_iou),
        soft_active_error_mean=safe_mean(active_error),
        weight_rmse_scaled=safe_mean(weight_rmse),
        teacher_target_peak_min_mean_db=safe_mean(teacher_weak),
        model_target_peak_min_mean_db=safe_mean(model_weak),
        delta_weak_peak_mean_db=safe_mean(model_weak - teacher_weak),
        teacher_psll_weak_mean_db=safe_mean(teacher_psll),
        model_psll_weak_mean_db=safe_mean(model_psll),
        delta_psll_weak_mean_db=safe_mean(model_psll - teacher_psll),
        teacher_psll_weak_p95_db=safe_percentile(teacher_psll, 95),
        model_psll_weak_p95_db=safe_percentile(model_psll, 95),
        teacher_isolation_worst_p05_db=safe_percentile(teacher_iso, 5),
        model_isolation_worst_p05_db=safe_percentile(model_iso, 5),
    )


def evaluate_model(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    features: np.ndarray,
    splits: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[list[EvalSummary], dict[str, float]]:
    all_indices = np.arange(arrays["k_values"].shape[0], dtype=np.int64)
    dataset = HfssArrayDataset(arrays, features, all_indices)
    pred = collect_predictions(model, dataset, device, batch_size=batch_size)
    order = np.argsort(pred["index"])
    mask_logits = pred["mask_logits"][order]
    weights_scaled_pred = pred["weights_scaled"][order]
    soft_active = pred["soft_active"][order]

    num_active = arrays["num_active"].astype(np.float32)
    pred_gate = hard_gate_from_logits(mask_logits, num_active)
    target_gate = arrays["masks"].astype(np.float32)
    intersection = (pred_gate * target_gate).sum(axis=1)
    union = np.maximum(pred_gate + target_gate, 0.0).clip(0.0, 1.0).sum(axis=1)
    mask_iou = intersection / np.maximum(union, 1.0)

    task_valid = arrays["task_valid"].astype(np.float32)
    pred_weights = (
        (weights_scaled_pred[..., 0] + 1j * weights_scaled_pred[..., 1])
        * pred_gate[:, :, None]
        * task_valid[:, None, :]
        / num_active[:, None, None]
    ).astype(np.complex64)
    teacher_weights = (
        arrays["task_weights_real_imag"][..., 0] + 1j * arrays["task_weights_real_imag"][..., 1]
    ).astype(np.complex64)
    target_weights_scaled = arrays["task_weights_real_imag"].astype(np.float32) * num_active[:, None, None, None]
    weight_diff = (weights_scaled_pred - target_weights_scaled) * task_valid[:, None, :, None]
    weight_rmse = np.sqrt((weight_diff**2).sum(axis=(1, 2, 3)) / np.maximum(task_valid.sum(axis=1) * NUM_ELEMENTS * 2, 1.0))

    grid_dirs, _ = build_eval_grid()
    teacher_metrics = compute_af_metrics(
        teacher_weights,
        arrays["targets_deg"].astype(np.float32),
        task_valid,
        arrays["positions_lambda"].astype(np.float32),
        grid_dirs,
    )
    model_metrics = compute_af_metrics(
        pred_weights,
        arrays["targets_deg"].astype(np.float32),
        task_valid,
        arrays["positions_lambda"].astype(np.float32),
        grid_dirs,
    )

    eval_arrays = {
        "teacher": teacher_metrics,
        "model": model_metrics,
        "mask_iou": mask_iou.astype(np.float32),
        "soft_active": soft_active.astype(np.float32),
        "active_ratio": arrays["active_ratios_requested"].astype(np.float32),
        "weight_rmse_scaled": weight_rmse.astype(np.float32),
    }
    rows: list[EvalSummary] = []
    split_by_index = np.empty(arrays["k_values"].shape[0], dtype=object)
    for split, idx in splits.items():
        split_by_index[idx] = split
        rows.append(summarize_group(split, "all", "all", idx, eval_arrays))
        for k in K_VALUES:
            k_idx = idx[arrays["k_values"][idx] == k]
            rows.append(summarize_group(split, k, "all", k_idx, eval_arrays))
            for ar in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                mask = np.isclose(arrays["active_ratios_requested"][k_idx], ar)
                rows.append(summarize_group(split, k, f"{ar:.1f}", k_idx[mask], eval_arrays))

    test_idx = splits["test"]
    k46_test = test_idx[np.isin(arrays["k_values"][test_idx], [4, 6])]
    headline = {
        "test_mask_iou_mean": safe_mean(mask_iou[test_idx]),
        "test_soft_active_error_mean": safe_mean(np.abs(soft_active[test_idx] - arrays["active_ratios_requested"][test_idx])),
        "test_weight_rmse_scaled": safe_mean(weight_rmse[test_idx]),
        "test_delta_psll_weak_mean_db": safe_mean(
            model_metrics["psll_to_weakest_peak_db"][test_idx] - teacher_metrics["psll_to_weakest_peak_db"][test_idx]
        ),
        "test_delta_weak_peak_mean_db": safe_mean(
            model_metrics["target_peak_min_db"][test_idx] - teacher_metrics["target_peak_min_db"][test_idx]
        ),
        "test_k46_delta_psll_weak_mean_db": safe_mean(
            model_metrics["psll_to_weakest_peak_db"][k46_test] - teacher_metrics["psll_to_weakest_peak_db"][k46_test]
        ),
        "test_k46_delta_weak_peak_mean_db": safe_mean(
            model_metrics["target_peak_min_db"][k46_test] - teacher_metrics["target_peak_min_db"][k46_test]
        ),
    }
    return rows, headline


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset_dir: Path = args.dataset_dir
    out_dir = args.out_dir
    if out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = dataset_dir / "training_runs" / f"surrogate_{stamp}"
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = SurrogateNet(features.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.08)

    _, _, train_grid_dirs_np = make_grid(theta_step=4.0, phi_step=8.0)
    train_grid_dirs = torch.from_numpy(train_grid_dirs_np).to(device)
    positions = torch.from_numpy(arrays["positions_lambda"].astype(np.float32)).to(device)

    config = {
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "device": str(device),
        "feature_dim": int(features.shape[1]),
        "num_train": int(splits["train"].shape[0]),
        "num_val": int(splits["val"].shape[0]),
        "num_test": int(splits["test"].shape[0]),
        "args": jsonable_args(args),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    log_rows: list[dict[str, object]] = []
    print(f"Training surrogate on {device}; output={out_dir}")
    print(f"splits train/val/test={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    start = time.time()
    for epoch in range(args.epochs):
        stats = train_one_epoch(model, train_loader, optimizer, device, positions, train_grid_dirs, epoch, args)
        scheduler.step()
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **stats}
        log_rows.append(row)
        if epoch == 0 or (epoch + 1) % args.train_eval_every == 0 or epoch + 1 == args.epochs:
            msg = ", ".join(f"{k}={v:.5g}" for k, v in stats.items())
            print(f"epoch {epoch + 1:03d}/{args.epochs}: {msg}")
            write_csv(out_dir / "train_log.csv", log_rows)

    torch.save({"model_state": model.state_dict(), "config": config}, out_dir / "surrogate_model.pt")

    print("Evaluating model on train/val/test...")
    summary_rows, headline = evaluate_model(model, arrays, features, splits, device, batch_size=args.batch_size)
    summary_dict_rows = [asdict(row) for row in summary_rows]
    write_csv(out_dir / "eval_summary_by_split_k_active.csv", summary_dict_rows)
    metrics = {
        "headline": headline,
        "elapsed_s": time.time() - start,
        "outputs": {
            "train_log": str(out_dir / "train_log.csv"),
            "eval_summary": str(out_dir / "eval_summary_by_split_k_active.csv"),
            "model": str(out_dir / "surrogate_model.pt"),
        },
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
