"""Train a scene-split, uncertainty-aware full-wave residual critic.

The model uses a 16x16 spatial encoder for mask/complex weights, a DeepSets
target encoder, scalar AF/conditioning features, heteroscedastic residual
regression, four calibrated feasibility heads, and scene-level listwise ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = (
    ROOT
    / "hfss_outputs"
    / "multitask_dataset"
    / "stage1_fullwave_residual_dataset_v2_20260714"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "hfss_outputs"
    / "multitask_dataset"
    / "training_runs"
    / "fullwave_residual_critic_v2_20260714"
)
KMAX = 6
NUM_ELEMENTS = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--scene-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=7.5e-4)
    parser.add_argument("--weight-decay", type=float, default=2.0e-4)
    parser.add_argument("--hard-negative-weight", type=float, default=3.0)
    parser.add_argument("--gate-loss-weight", type=float, default=0.60)
    parser.add_argument("--rank-loss-weight", type=float, default=0.25)
    parser.add_argument("--rank-temperature", type=float, default=0.75)
    parser.add_argument("--uncertainty-kappa", type=float, default=1.0)
    parser.add_argument("--seeds", default="20260714,20260715,20260716,20260717,20260718")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    arrays = np.load(path, allow_pickle=False)
    return {key: arrays[key] for key in arrays.files}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def target_features(targets_deg: np.ndarray, task_valid: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(np.nan_to_num(targets_deg[..., 0], nan=0.0))
    phi = np.deg2rad(np.nan_to_num(targets_deg[..., 1], nan=0.0))
    valid = task_valid.astype(np.float32)
    features = np.stack(
        [
            valid,
            (np.rad2deg(theta) / 90.0) * valid,
            np.sin(phi) * valid,
            np.cos(phi) * valid,
            np.sin(theta) * np.cos(phi) * valid,
            np.sin(theta) * np.sin(phi) * valid,
            np.cos(theta) * valid,
        ],
        axis=-1,
    )
    return features.astype(np.float32)


def spatial_features(
    masks: np.ndarray,
    weights_ri: np.ndarray,
    num_active: np.ndarray,
) -> np.ndarray:
    n = masks.shape[0]
    weight_scaled = weights_ri.astype(np.float32) * num_active[:, None, None, None].clip(min=1.0)
    weight_channels = weight_scaled.transpose(0, 2, 3, 1).reshape(n, KMAX * 2, 16, 16)
    mask_channel = masks.reshape(n, 1, 16, 16).astype(np.float32)
    return np.concatenate([mask_channel, weight_channels], axis=1).astype(np.float32)


def standardize(
    values: np.ndarray,
    train_indices: np.ndarray,
    axes: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = values[train_indices]
    if axes is None:
        mean = train.mean(axis=0, keepdims=True)
        std = train.std(axis=0, keepdims=True)
    else:
        mean = train.mean(axis=axes, keepdims=True)
        std = train.std(axis=axes, keepdims=True)
    std = np.where(std < 1.0e-6, 1.0, std)
    return ((values - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


class ResidualDataset(Dataset):
    def __init__(
        self,
        spatial: np.ndarray,
        targets: np.ndarray,
        scalars: np.ndarray,
        residuals: np.ndarray,
        gates: np.ndarray,
        rank_violation: np.ndarray,
        sample_index: np.ndarray,
        hard_negative: np.ndarray,
        indices: np.ndarray,
    ):
        self.spatial = spatial
        self.targets = targets
        self.scalars = scalars
        self.residuals = residuals
        self.gates = gates
        self.rank_violation = rank_violation
        self.sample_index = sample_index
        self.hard_negative = hard_negative
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        return {
            "row_index": torch.tensor(idx, dtype=torch.long),
            "spatial": torch.from_numpy(self.spatial[idx]),
            "targets": torch.from_numpy(self.targets[idx]),
            "scalars": torch.from_numpy(self.scalars[idx]),
            "residuals": torch.from_numpy(self.residuals[idx]),
            "gates": torch.from_numpy(self.gates[idx]),
            "rank_violation": torch.tensor(self.rank_violation[idx], dtype=torch.float32),
            "sample_index": torch.tensor(self.sample_index[idx], dtype=torch.long),
            "hard_negative": torch.tensor(self.hard_negative[idx], dtype=torch.float32),
        }


class SceneBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: ResidualDataset,
        scene_batch_size: int,
        *,
        shuffle: bool,
        seed: int,
    ):
        self.dataset = dataset
        self.scene_batch_size = max(1, int(scene_batch_size))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        local_by_scene: dict[int, list[int]] = defaultdict(list)
        for local_index, global_index in enumerate(dataset.indices.tolist()):
            local_by_scene[int(dataset.sample_index[global_index])].append(local_index)
        self.local_by_scene = local_by_scene
        self.scenes = sorted(local_by_scene)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        scenes = np.asarray(self.scenes, dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(scenes)
        for start in range(0, scenes.size, self.scene_batch_size):
            batch: list[int] = []
            for scene in scenes[start : start + self.scene_batch_size].tolist():
                batch.extend(self.local_by_scene[int(scene)])
            yield batch

    def __len__(self) -> int:
        return int(math.ceil(len(self.scenes) / self.scene_batch_size))


class FullwaveResidualCritic(nn.Module):
    def __init__(self, scalar_dim: int, residual_dim: int, gate_dim: int):
        super().__init__()
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(13, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(6, 48),
            nn.SiLU(),
            nn.Conv2d(48, 72, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 72),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(72, 96),
            nn.SiLU(),
        )
        self.target_item = nn.Sequential(
            nn.Linear(7, 48),
            nn.SiLU(),
            nn.Linear(48, 64),
            nn.SiLU(),
        )
        self.target_project = nn.Sequential(nn.Linear(128, 96), nn.SiLU())
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(96 + 96 + 64, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(256, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
        )
        self.residual_mean = nn.Linear(160, residual_dim)
        self.residual_logvar = nn.Linear(160, residual_dim)
        self.gate_head = nn.Linear(160, gate_dim)
        self.rank_head = nn.Linear(160, 1)

    def forward(
        self,
        spatial: torch.Tensor,
        targets: torch.Tensor,
        scalars: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial_embedding = self.spatial_encoder(spatial)
        item_embedding = self.target_item(targets)
        valid = targets[..., :1]
        denom = valid.sum(dim=1).clamp_min(1.0)
        target_mean = (item_embedding * valid).sum(dim=1) / denom
        masked = item_embedding.masked_fill(valid <= 0.0, -1.0e4)
        target_max = masked.max(dim=1).values
        target_max = torch.where(torch.isfinite(target_max), target_max, torch.zeros_like(target_max))
        target_embedding = self.target_project(torch.cat([target_mean, target_max], dim=1))
        scalar_embedding = self.scalar_encoder(scalars)
        hidden = self.trunk(torch.cat([spatial_embedding, target_embedding, scalar_embedding], dim=1))
        mean = self.residual_mean(hidden)
        logvar = self.residual_logvar(hidden).clamp(-6.0, 5.0)
        gates = self.gate_head(hidden)
        rank = self.rank_head(hidden).squeeze(-1)
        return mean, logvar, gates, rank


def heteroscedastic_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    row_weight: torch.Tensor,
) -> torch.Tensor:
    nll = 0.5 * (torch.exp(-logvar) * (mean - target) ** 2 + logvar)
    nll = nll.mean(dim=1) * row_weight
    return nll.sum() / row_weight.sum().clamp_min(1.0)


def gate_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    row_weight: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    loss = loss.mean(dim=1) * row_weight
    return loss.sum() / row_weight.sum().clamp_min(1.0)


def listwise_loss(
    rank_score: torch.Tensor,
    rank_violation: torch.Tensor,
    sample_index: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for scene in torch.unique(sample_index):
        mask = sample_index == scene
        if int(mask.sum()) < 2:
            continue
        target_distribution = torch.softmax(-rank_violation[mask] / max(float(temperature), 1.0e-3), dim=0)
        log_prob = torch.log_softmax(rank_score[mask], dim=0)
        losses.append(-(target_distribution * log_prob).sum())
    if not losses:
        return rank_score.sum() * 0.0
    return torch.stack(losses).mean()


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not positives.size or not negatives.size:
        return float("nan")
    comparisons = positives[:, None] - negatives[None, :]
    return float(np.mean(comparisons > 0.0) + 0.5 * np.mean(comparisons == 0.0))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    precision = cumulative / np.arange(1, labels.size + 1)
    return float(np.sum(precision * sorted_labels) / positives)


def calibration_bins(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bin_index in range(bins):
        low = bin_index / bins
        high = (bin_index + 1) / bins
        mask = (probabilities >= low) & (
            probabilities <= high if bin_index == bins - 1 else probabilities < high
        )
        rows.append(
            {
                "bin": bin_index,
                "low": low,
                "high": high,
                "count": int(mask.sum()),
                "mean_probability": float(probabilities[mask].mean()) if np.any(mask) else float("nan"),
                "positive_rate": float(labels[mask].mean()) if np.any(mask) else float("nan"),
            }
        )
    return rows


def binary_metrics(labels: np.ndarray, logits: np.ndarray, temperature: float = 1.0) -> dict[str, float]:
    labels = labels.astype(np.int64)
    probabilities = sigmoid(logits / max(float(temperature), 1.0e-4))
    predicted = probabilities >= 0.5
    tp = int(np.sum(predicted & (labels == 1)))
    tn = int(np.sum((~predicted) & (labels == 0)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum((~predicted) & (labels == 1)))
    rows = calibration_bins(labels, probabilities)
    ece = sum(
        (row["count"] / max(labels.size, 1)) * abs(row["mean_probability"] - row["positive_rate"])
        for row in rows
        if row["count"] > 0
    )
    eps = 1.0e-8
    nll = -float(
        np.mean(labels * np.log(probabilities + eps) + (1 - labels) * np.log(1.0 - probabilities + eps))
    )
    return {
        "positive_rate": float(labels.mean()),
        "auroc": binary_auc(labels, probabilities),
        "auprc": average_precision(labels, probabilities),
        "accuracy": float(np.mean(predicted == labels)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "ece": float(ece),
        "nll": nll,
    }


def fit_temperature(labels: np.ndarray, logits: np.ndarray) -> float:
    labels = labels.astype(np.float64)
    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in np.geomspace(0.20, 5.0, 241):
        probabilities = sigmoid(logits / temperature)
        eps = 1.0e-8
        nll = -float(
            np.mean(labels * np.log(probabilities + eps) + (1.0 - labels) * np.log(1.0 - probabilities + eps))
        )
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature)
    return best_temperature


def predict(
    model: FullwaveResidualCritic,
    dataset: ResidualDataset,
    device: torch.device,
    scene_batch_size: int,
) -> dict[str, np.ndarray]:
    sampler = SceneBatchSampler(dataset, scene_batch_size, shuffle=False, seed=0)
    loader = DataLoader(dataset, batch_sampler=sampler)
    model.eval()
    output: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            mean, logvar, gates, rank = model(
                batch["spatial"].to(device),
                batch["targets"].to(device),
                batch["scalars"].to(device),
            )
            output["row_index"].append(batch["row_index"].numpy())
            output["mean"].append(mean.cpu().numpy())
            output["logvar"].append(logvar.cpu().numpy())
            output["gates"].append(gates.cpu().numpy())
            output["rank"].append(rank.cpu().numpy())
    return {key: np.concatenate(parts, axis=0) for key, parts in output.items()}


def flatten_metrics(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in values.items()}


def selection_metrics(
    *,
    indices: np.ndarray,
    data: dict[str, np.ndarray],
    pred_residual: np.ndarray,
    pred_sigma: np.ndarray,
    gate_logits: np.ndarray,
    rank_score: np.ndarray,
    temperatures: np.ndarray,
    kappa: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    gate_names = [str(name) for name in data["gate_names"]]
    gate_col = {name: idx for idx, name in enumerate(gate_names)}
    residual_names = [str(name) for name in data["residual_names"]]
    residual_col = {name: idx for idx, name in enumerate(residual_names)}
    scalar_names = [str(name) for name in data["scalar_names"]]
    scalar_col = {name: idx for idx, name in enumerate(scalar_names)}
    probabilities = sigmoid(gate_logits / temperatures[None, :])
    predicted_hfss = data["af_metrics"][:, :5] + pred_residual
    by_scene: dict[int, list[int]] = defaultdict(list)
    for idx in indices.tolist():
        by_scene[int(data["sample_index"][idx])].append(idx)
    selected_rows: list[dict[str, Any]] = []
    methods = ("rank", "gate15_prob", "gate20_prob", "conservative")
    selected_by_method: dict[str, list[int]] = {name: [] for name in methods}
    oracle15: list[float] = []
    oracle20: list[float] = []
    oracle_engineering: list[float] = []
    top2_gate20: list[float] = []
    for sample, scene_indices in sorted(by_scene.items()):
        scene = np.asarray(scene_indices, dtype=np.int64)
        gate15_probability = probabilities[scene, gate_col["gate15"]]
        gate20_probability = probabilities[scene, gate_col["gate20"]]
        main_probability = probabilities[scene, gate_col["mainlobe_gate"]]
        chosen_rank = int(scene[int(np.argmax(rank_score[scene]))])
        chosen_gate15 = int(scene[int(np.argmax(gate15_probability))])
        chosen_gate20 = int(scene[int(np.argmax(gate20_probability))])
        psll_ucb = (
            predicted_hfss[scene, residual_col["psll_db"]]
            + float(kappa) * pred_sigma[scene, residual_col["psll_db"]]
        )
        nearest_lcb = (
            predicted_hfss[scene, residual_col["iso_nearest_db"]]
            - float(kappa) * pred_sigma[scene, residual_col["iso_nearest_db"]]
        )
        local_lcb = (
            predicted_hfss[scene, residual_col["iso_local_db"]]
            - float(kappa) * pred_sigma[scene, residual_col["iso_local_db"]]
        )
        predicted_peak = predicted_hfss[scene, residual_col["peak_min_db"]]
        predicted_spread = predicted_hfss[scene, residual_col["peak_spread_db"]]
        reference_peak = data["scalar_features"][scene, scalar_col["reference_hfss_peak_min_db"]]
        predicted_drop = reference_peak - predicted_peak
        feasible = (
            (psll_ucb <= 0.0)
            & (nearest_lcb >= 25.0)
            & (local_lcb >= 20.0)
            & (predicted_drop <= 0.5)
            & (predicted_spread <= 3.0)
            & (main_probability >= 0.5)
        )
        conservative_utility = (
            2.0 * gate20_probability
            + main_probability
            - np.maximum(psll_ucb, 0.0)
            - 0.5 * np.maximum(25.0 - nearest_lcb, 0.0)
            - 0.35 * np.maximum(20.0 - local_lcb, 0.0)
            - 0.25 * np.maximum(predicted_drop - 0.5, 0.0)
            - 0.15 * np.maximum(predicted_spread - 3.0, 0.0)
        )
        if np.any(feasible):
            local_choice = np.flatnonzero(feasible)[int(np.argmax(conservative_utility[feasible]))]
        else:
            local_choice = int(np.argmax(conservative_utility))
        chosen_conservative = int(scene[local_choice])
        chosen = {
            "rank": chosen_rank,
            "gate15_prob": chosen_gate15,
            "gate20_prob": chosen_gate20,
            "conservative": chosen_conservative,
        }
        for method, row_index in chosen.items():
            selected_by_method[method].append(row_index)
            selected_rows.append(
                {
                    "sample_index": sample,
                    "method": method,
                    "row_index": row_index,
                    "strategy": str(data["strategy"][row_index]),
                    "active_ratio": float(data["active_ratios"][row_index]),
                    "gate15_probability": float(probabilities[row_index, gate_col["gate15"]]),
                    "gate20_probability": float(probabilities[row_index, gate_col["gate20"]]),
                    "mainlobe_probability": float(probabilities[row_index, gate_col["mainlobe_gate"]]),
                    "actual_gate15": int(data["gates"][row_index, gate_col["gate15"]]),
                    "actual_gate20": int(data["gates"][row_index, gate_col["gate20"]]),
                    "actual_mainlobe_gate": int(data["gates"][row_index, gate_col["mainlobe_gate"]]),
                    "actual_strict_engineering_gate": int(
                        data["gates"][row_index, gate_col["strict_engineering_gate"]]
                    ),
                    "rank_score": float(rank_score[row_index]),
                }
            )
        actual = data["gates"][scene]
        oracle15.append(float(np.any(actual[:, gate_col["gate15"]] >= 0.5)))
        oracle20.append(float(np.any(actual[:, gate_col["gate20"]] >= 0.5)))
        oracle_engineering.append(float(np.any(actual[:, gate_col["strict_engineering_gate"]] >= 0.5)))
        top2 = scene[np.argsort(-gate20_probability)[:2]]
        top2_gate20.append(float(np.any(data["gates"][top2, gate_col["gate20"]] >= 0.5)))

    summary: dict[str, float] = {
        "scene_count": float(len(by_scene)),
        "oracle_gate15_rate": float(np.mean(oracle15)),
        "oracle_gate20_rate": float(np.mean(oracle20)),
        "oracle_strict_engineering_rate": float(np.mean(oracle_engineering)),
        "top2_gate20_probability_rate": float(np.mean(top2_gate20)),
    }
    for method, selected in selected_by_method.items():
        chosen = np.asarray(selected, dtype=np.int64)
        for gate_name, col in gate_col.items():
            summary[f"{method}_{gate_name}_rate"] = float(data["gates"][chosen, col].mean())
    return summary, selected_rows


def calibration_svg(path: Path, rows: list[dict[str, Any]], gate_names: list[str]) -> None:
    width, height = 720, 520
    left, top, plot = 70, 45, 400
    colors = ["#1565c0", "#c62828", "#2e7d32", "#6a1b9a"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top + plot}" x2="{left + plot}" y2="{top}" stroke="#777" stroke-dasharray="5 5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top + plot}" x2="{left + plot}" y2="{top + plot}" stroke="#222"/>',
        '<text x="270" y="505" font-family="Arial" font-size="14">Mean predicted probability</text>',
        '<text x="18" y="275" font-family="Arial" font-size="14" transform="rotate(-90 18 275)">Observed positive rate</text>',
        '<text x="70" y="25" font-family="Arial" font-size="18">Scene-test calibration</text>',
    ]
    for tick in range(6):
        value = tick / 5.0
        x = left + value * plot
        y = top + plot - value * plot
        lines.append(f'<text x="{x - 8:.1f}" y="{top + plot + 22}" font-family="Arial" font-size="11">{value:.1f}</text>')
        lines.append(f'<text x="{left - 34}" y="{y + 4:.1f}" font-family="Arial" font-size="11">{value:.1f}</text>')
    for gate_index, gate_name in enumerate(gate_names):
        gate_rows = [row for row in rows if row["gate"] == gate_name and row["count"] > 0]
        points = []
        for row in gate_rows:
            x = left + float(row["mean_probability"]) * plot
            y = top + plot - float(row["positive_rate"]) * plot
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            color = colors[gate_index % len(colors)]
            lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for point in points:
                x, y = point.split(",")
                lines.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
            legend_y = 70 + 24 * gate_index
            lines.append(f'<line x1="500" y1="{legend_y}" x2="530" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
            lines.append(f'<text x="540" y="{legend_y + 5}" font-family="Arial" font-size="13">{gate_name}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def train_one_seed(
    *,
    seed: int,
    args: argparse.Namespace,
    data: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    set_seed(seed)
    seed_dir = args.out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    train_indices = np.flatnonzero(data["split_id"] == 0)
    val_indices = np.flatnonzero(data["split_id"] == 1)
    test_indices = np.flatnonzero(data["split_id"] == 2)
    spatial_raw = spatial_features(data["masks"], data["weights_real_imag"], data["num_active"])
    spatial = spatial_raw.copy()
    weight_std, spatial_mean, spatial_scale = standardize(spatial_raw[:, 1:], train_indices, axes=(0, 2, 3))
    spatial[:, 1:] = weight_std
    targets = target_features(data["targets_deg"], data["task_valid"])
    scalars, scalar_mean, scalar_scale = standardize(data["scalar_features"], train_indices)
    residuals, residual_mean, residual_scale = standardize(data["residuals"], train_indices)
    datasets = {
        "train": ResidualDataset(
            spatial,
            targets,
            scalars,
            residuals,
            data["gates"],
            data["rank_violation"],
            data["sample_index"],
            data["hard_negative"],
            train_indices,
        ),
        "val": ResidualDataset(
            spatial,
            targets,
            scalars,
            residuals,
            data["gates"],
            data["rank_violation"],
            data["sample_index"],
            data["hard_negative"],
            val_indices,
        ),
        "test": ResidualDataset(
            spatial,
            targets,
            scalars,
            residuals,
            data["gates"],
            data["rank_violation"],
            data["sample_index"],
            data["hard_negative"],
            test_indices,
        ),
    }
    train_sampler = SceneBatchSampler(
        datasets["train"],
        int(args.scene_batch_size),
        shuffle=True,
        seed=seed,
    )
    train_loader = DataLoader(datasets["train"], batch_sampler=train_sampler)
    gate_positive_rate = data["gates"][train_indices].mean(axis=0)
    pos_weight = np.clip((1.0 - gate_positive_rate) / np.maximum(gate_positive_rate, 1.0e-4), 1.0, 25.0)
    pos_weight_tensor = torch.from_numpy(pos_weight.astype(np.float32)).to(device)
    model = FullwaveResidualCritic(
        scalar_dim=data["scalar_features"].shape[1],
        residual_dim=data["residuals"].shape[1],
        gate_dim=data["gates"].shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(args.epochs), 1))
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    log_rows: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        totals = defaultdict(float)
        examples = 0
        for batch in train_loader:
            spatial_batch = batch["spatial"].to(device)
            targets_batch = batch["targets"].to(device)
            scalars_batch = batch["scalars"].to(device)
            residual_batch = batch["residuals"].to(device)
            gates_batch = batch["gates"].to(device)
            hard_negative = batch["hard_negative"].to(device)
            row_weight = 1.0 + hard_negative * (float(args.hard_negative_weight) - 1.0)
            mean, logvar, gate_logits, rank_score = model(spatial_batch, targets_batch, scalars_batch)
            loss_residual = heteroscedastic_loss(mean, logvar, residual_batch, row_weight)
            loss_gate = gate_loss(gate_logits, gates_batch, row_weight, pos_weight_tensor)
            loss_rank = listwise_loss(
                rank_score,
                batch["rank_violation"].to(device),
                batch["sample_index"].to(device),
                float(args.rank_temperature),
            )
            loss = (
                loss_residual
                + float(args.gate_loss_weight) * loss_gate
                + float(args.rank_loss_weight) * loss_rank
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = int(spatial_batch.shape[0])
            examples += count
            totals["loss"] += float(loss.detach().cpu()) * count
            totals["residual"] += float(loss_residual.detach().cpu()) * count
            totals["gate"] += float(loss_gate.detach().cpu()) * count
            totals["rank"] += float(loss_rank.detach().cpu()) * count
        scheduler.step()
        val_prediction = predict(model, datasets["val"], device, int(args.scene_batch_size))
        val_rows = val_prediction["row_index"].astype(np.int64)
        val_residual_mae = float(
            np.mean(
                np.abs(
                    val_prediction["mean"] * residual_scale.reshape(1, -1)
                    + residual_mean.reshape(1, -1)
                    - data["residuals"][val_rows]
                )
            )
        )
        val_gate20 = binary_metrics(data["gates"][val_rows, 1], val_prediction["gates"][:, 1])
        val_main = binary_metrics(data["gates"][val_rows, 2], val_prediction["gates"][:, 2])
        score = val_gate20["nll"] + 0.35 * val_main["nll"] + 0.03 * val_residual_mae
        if score < best_score - 1.0e-5:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        log_row = {
            "epoch": epoch,
            "loss": totals["loss"] / max(examples, 1),
            "residual_loss": totals["residual"] / max(examples, 1),
            "gate_loss": totals["gate"] / max(examples, 1),
            "rank_loss": totals["rank"] / max(examples, 1),
            "val_residual_mae_db": val_residual_mae,
            "val_gate20_auroc": val_gate20["auroc"],
            "val_gate20_auprc": val_gate20["auprc"],
            "val_gate20_ece_raw": val_gate20["ece"],
            "val_mainlobe_auroc": val_main["auroc"],
            "selection_score": score,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": time.time() - started,
        }
        log_rows.append(log_row)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({"seed": seed, **log_row}))
        if stale >= int(args.patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    predictions = {name: predict(model, dataset, device, int(args.scene_batch_size)) for name, dataset in datasets.items()}
    gate_names = [str(name) for name in data["gate_names"]]
    residual_names = [str(name) for name in data["residual_names"]]
    val_rows = predictions["val"]["row_index"].astype(np.int64)
    temperatures = np.asarray(
        [
            fit_temperature(data["gates"][val_rows, gate_col], predictions["val"]["gates"][:, gate_col])
            for gate_col in range(len(gate_names))
        ],
        dtype=np.float32,
    )
    eval_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selection_by_split: dict[str, dict[str, float]] = {}
    selected_rows_all: list[dict[str, Any]] = []
    for split_name, prediction in predictions.items():
        rows = prediction["row_index"].astype(np.int64)
        pred_residual = prediction["mean"] * residual_scale.reshape(1, -1) + residual_mean.reshape(1, -1)
        pred_sigma = np.exp(0.5 * prediction["logvar"]) * residual_scale.reshape(1, -1)
        split_summary: dict[str, Any] = {
            "split": split_name,
            "n": int(rows.size),
            "scene_count": int(np.unique(data["sample_index"][rows]).size),
        }
        for metric_col, metric_name in enumerate(residual_names):
            error = pred_residual[:, metric_col] - data["residuals"][rows, metric_col]
            split_summary[f"residual_{metric_name}_mae"] = float(np.mean(np.abs(error)))
            split_summary[f"residual_{metric_name}_rmse"] = float(np.sqrt(np.mean(error**2)))
            split_summary[f"residual_{metric_name}_coverage_1sigma"] = float(
                np.mean(np.abs(error) <= pred_sigma[:, metric_col])
            )
        for gate_col, gate_name in enumerate(gate_names):
            raw = binary_metrics(data["gates"][rows, gate_col], prediction["gates"][:, gate_col])
            calibrated = binary_metrics(
                data["gates"][rows, gate_col],
                prediction["gates"][:, gate_col],
                float(temperatures[gate_col]),
            )
            split_summary.update(flatten_metrics(f"{gate_name}_raw", raw))
            split_summary.update(flatten_metrics(f"{gate_name}_cal", calibrated))
            probabilities = sigmoid(prediction["gates"][:, gate_col] / float(temperatures[gate_col]))
            if split_name == "test":
                for row in calibration_bins(data["gates"][rows, gate_col], probabilities):
                    calibration_rows.append({"gate": gate_name, **row})
        full_pred_residual = np.zeros_like(data["residuals"], dtype=np.float32)
        full_pred_sigma = np.zeros_like(data["residuals"], dtype=np.float32)
        full_gate_logits = np.zeros_like(data["gates"], dtype=np.float32)
        full_rank_score = np.zeros(data["sample_index"].shape[0], dtype=np.float32)
        full_pred_residual[rows] = pred_residual
        full_pred_sigma[rows] = pred_sigma
        full_gate_logits[rows] = prediction["gates"]
        full_rank_score[rows] = prediction["rank"]
        selection, selected_rows = selection_metrics(
            indices=rows,
            data=data,
            pred_residual=full_pred_residual,
            pred_sigma=full_pred_sigma,
            gate_logits=full_gate_logits,
            rank_score=full_rank_score,
            temperatures=temperatures,
            kappa=float(args.uncertainty_kappa),
        )
        split_summary.update(selection)
        selection_by_split[split_name] = selection
        for selected in selected_rows:
            selected_rows_all.append({"split": split_name, **selected})
        eval_rows.append(split_summary)
        calibrated_probabilities = sigmoid(prediction["gates"] / temperatures[None, :])
        for local_pos, row_index in enumerate(rows.tolist()):
            record: dict[str, Any] = {
                "row_index": row_index,
                "split": split_name,
                "sample_index": int(data["sample_index"][row_index]),
                "strategy": str(data["strategy"][row_index]),
                "active_ratio": float(data["active_ratios"][row_index]),
                "rank_score": float(prediction["rank"][local_pos]),
            }
            for metric_col, metric_name in enumerate(residual_names):
                record[f"true_residual_{metric_name}"] = float(data["residuals"][row_index, metric_col])
                record[f"pred_residual_{metric_name}"] = float(pred_residual[local_pos, metric_col])
                record[f"pred_sigma_{metric_name}"] = float(pred_sigma[local_pos, metric_col])
            for gate_col, gate_name in enumerate(gate_names):
                record[f"true_{gate_name}"] = int(data["gates"][row_index, gate_col])
                record[f"prob_{gate_name}"] = float(calibrated_probabilities[local_pos, gate_col])
            prediction_rows.append(record)

    write_csv(seed_dir / "train_log.csv", log_rows)
    write_csv(seed_dir / "eval_summary.csv", eval_rows)
    write_csv(seed_dir / "predictions.csv", prediction_rows)
    write_csv(seed_dir / "candidate_selection.csv", selected_rows_all)
    write_csv(seed_dir / "calibration_curve.csv", calibration_rows)
    calibration_svg(seed_dir / "calibration_curve.svg", calibration_rows, gate_names)
    checkpoint = {
        "model_state": model.state_dict(),
        "scalar_dim": int(data["scalar_features"].shape[1]),
        "residual_dim": int(data["residuals"].shape[1]),
        "gate_dim": int(data["gates"].shape[1]),
        "spatial_mean": spatial_mean,
        "spatial_scale": spatial_scale,
        "scalar_mean": scalar_mean,
        "scalar_scale": scalar_scale,
        "residual_mean": residual_mean,
        "residual_scale": residual_scale,
        "temperatures": temperatures,
        "residual_names": residual_names,
        "gate_names": gate_names,
        "seed": seed,
        "args": vars(args),
    }
    torch.save(checkpoint, seed_dir / "residual_critic_v2.pt")
    test_summary = next(row for row in eval_rows if row["split"] == "test")
    val_summary = next(row for row in eval_rows if row["split"] == "val")
    run_summary = {
        "seed": seed,
        "device": str(device),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "epochs_completed": len(log_rows),
        "best_selection_score": best_score,
        "temperatures": {name: float(temperatures[col]) for col, name in enumerate(gate_names)},
        "train_variant_count": int(train_indices.size),
        "val_variant_count": int(val_indices.size),
        "test_variant_count": int(test_indices.size),
        "val": val_summary,
        "test": test_summary,
        "selection": selection_by_split,
        "elapsed_s": time.time() - started,
        "outputs": {
            "checkpoint": str(seed_dir / "residual_critic_v2.pt"),
            "eval_summary": str(seed_dir / "eval_summary.csv"),
            "predictions": str(seed_dir / "predictions.csv"),
            "candidate_selection": str(seed_dir / "candidate_selection.csv"),
            "calibration_svg": str(seed_dir / "calibration_curve.svg"),
        },
    }
    (seed_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps({"seed_complete": seed, "test": test_summary}, indent=2))
    return run_summary


def mean_ci95(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, 0.0
    return mean, float(1.96 * arr.std(ddof=1) / math.sqrt(arr.size))


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(args.device))
    data = load_npz(args.dataset_dir / "fullwave_residual_dataset_v2.npz")
    seeds = [int(part.strip()) for part in str(args.seeds).split(",") if part.strip()]
    summaries = [train_one_seed(seed=seed, args=args, data=data, device=device) for seed in seeds]
    metric_paths = {
        "gate15_auroc": "gate15_cal_auroc",
        "gate15_auprc": "gate15_cal_auprc",
        "gate15_ece": "gate15_cal_ece",
        "gate20_auroc": "gate20_cal_auroc",
        "gate20_auprc": "gate20_cal_auprc",
        "gate20_ece": "gate20_cal_ece",
        "mainlobe_auroc": "mainlobe_gate_cal_auroc",
        "strict_engineering_auroc": "strict_engineering_gate_cal_auroc",
        "rank_gate15_rate": "rank_gate15_rate",
        "rank_gate20_rate": "rank_gate20_rate",
        "rank_strict_engineering_rate": "rank_strict_engineering_gate_rate",
        "conservative_gate15_rate": "conservative_gate15_rate",
        "conservative_gate20_rate": "conservative_gate20_rate",
        "conservative_strict_engineering_rate": "conservative_strict_engineering_gate_rate",
        "oracle_gate15_rate": "oracle_gate15_rate",
        "oracle_gate20_rate": "oracle_gate20_rate",
        "oracle_strict_engineering_rate": "oracle_strict_engineering_rate",
    }
    aggregate_rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {
        "seeds": seeds,
        "dataset_dir": str(args.dataset_dir),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "metrics": {},
    }
    for label, key in metric_paths.items():
        values = [float(summary["test"].get(key, float("nan"))) for summary in summaries]
        mean, ci = mean_ci95(values)
        aggregate_rows.append(
            {
                "metric": label,
                "mean": mean,
                "ci95_half_width": ci,
                **{f"seed_{seed}": value for seed, value in zip(seeds, values)},
            }
        )
        aggregate["metrics"][label] = {"mean": mean, "ci95_half_width": ci, "values": values}
    best_summary = max(
        summaries,
        key=lambda summary: float(summary["val"].get("gate20_cal_auroc", float("-inf"))),
    )
    aggregate["best_seed"] = int(best_summary["seed"])
    aggregate["run_summaries"] = [str(args.out_dir / f"seed_{seed}" / "run_summary.json") for seed in seeds]
    write_csv(args.out_dir / "five_seed_summary.csv", aggregate_rows)
    (args.out_dir / "five_seed_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
