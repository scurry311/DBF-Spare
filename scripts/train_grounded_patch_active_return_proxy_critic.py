"""Pretrain an active-return feasibility critic on the local-full-wave S256 proxy.

This checkpoint is intentionally namespaced as proxy pretraining.  It must not
be reported as a full-array HFSS critic until fine-tuned and tested on true
16x16 full-wave labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT
    / "hfss_outputs"
    / "grounded_patch_s256_proxy_20260717_run01"
    / "active_return_proxy_pretraining_dataset.npz"
)
DEFAULT_OUT = ROOT / "hfss_outputs" / "grounded_patch_proxy_critic_20260717_run01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260717, 20260718, 20260719, 20260720, 20260721])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ProxyDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray):
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        index = int(self.indices[item])
        mask = self.arrays["masks"][index].reshape(16, 16).astype(np.float32)
        weights = self.arrays["weights_real_imag"][index].reshape(16, 16, 2).astype(np.float32)
        amplitude = np.linalg.norm(weights, axis=-1)
        scale = max(float(np.max(amplitude)), 1.0e-8)
        spatial = np.stack((mask, weights[..., 0] / scale, weights[..., 1] / scale, amplitude / scale))
        k_value = int(self.arrays["k_values"][index])
        targets = np.zeros((6, 3), dtype=np.float32)
        targets[:k_value, 0] = self.arrays["targets_deg"][index, :k_value, 0] / 90.0
        targets[:k_value, 1] = self.arrays["targets_deg"][index, :k_value, 1] / 180.0
        targets[:k_value, 2] = 1.0
        scalars = np.asarray((k_value / 6.0, float(self.arrays["ratios"][index])), dtype=np.float32)
        regression = np.asarray(
            (
                np.clip(self.arrays["worst_active_rl_db"][index], -40.0, 40.0) / 20.0,
                np.clip(self.arrays["total_return_loss_db"][index], -40.0, 40.0) / 20.0,
            ),
            dtype=np.float32,
        )
        return (
            torch.from_numpy(spatial),
            torch.from_numpy(targets),
            torch.from_numpy(scalars),
            torch.tensor(float(self.arrays["proxy_gate"][index]), dtype=torch.float32),
            torch.from_numpy(regression),
            torch.tensor(index, dtype=torch.int64),
        )


class ActiveReturnCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.target_item = nn.Sequential(nn.Linear(3, 24), nn.SiLU(), nn.Linear(24, 32), nn.SiLU())
        self.trunk = nn.Sequential(
            nn.Linear(32 * 4 + 32 + 2, 128),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.SiLU(),
        )
        self.gate_head = nn.Linear(64, 1)
        self.regression_head = nn.Linear(64, 2)

    def forward(self, spatial: torch.Tensor, targets: torch.Tensor, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_features = self.spatial(spatial)
        items = self.target_item(targets)
        valid = targets[..., 2:3]
        target_features = (items * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        hidden = self.trunk(torch.cat((spatial_features, target_features, scalars), dim=1))
        return self.gate_head(hidden).squeeze(1), self.regression_head(hidden)


def split_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(labels.size)
    train, remainder = train_test_split(indices, test_size=0.30, random_state=seed, stratify=labels)
    val, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=seed + 1000,
        stratify=labels[remainder],
    )
    return np.sort(train), np.sort(val), np.sort(test)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    regressions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for spatial, targets, scalars, gate, _, index in loader:
        output, regression = model(spatial, targets, scalars)
        logits.append(output.numpy())
        regressions.append(regression.numpy())
        labels.append(gate.numpy())
        indices.append(index.numpy())
    return np.concatenate(logits), np.concatenate(regressions), np.concatenate(labels), np.concatenate(indices)


def calibrate_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logit_tensor = torch.tensor(logits, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=80, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = nn.functional.binary_cross_entropy_with_logits(logit_tensor / temperature, label_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def expected_calibration_error(probability: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (probability >= lower) & (probability < upper if upper < 1.0 else probability <= upper)
        if np.any(selected):
            result += float(np.mean(selected)) * abs(float(np.mean(probability[selected])) - float(np.mean(labels[selected])))
    return result


def metrics(logits: np.ndarray, labels: np.ndarray, temperature: float) -> dict[str, float]:
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -40.0, 40.0)))
    prediction = probability >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.astype(int), prediction.astype(int), average="binary", zero_division=0
    )
    return {
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "ece": expected_calibration_error(probability, labels),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate": float(np.mean(labels)),
        "predicted_positive_rate": float(np.mean(prediction)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_seed(args: argparse.Namespace, arrays: dict[str, np.ndarray], seed: int) -> dict[str, Any]:
    set_seed(seed)
    labels = arrays["proxy_gate"].astype(np.int64)
    train_indices, val_indices, test_indices = split_indices(labels, seed)
    loaders = {
        name: DataLoader(
            ProxyDataset(arrays, indices),
            batch_size=int(args.batch_size),
            shuffle=name == "train",
            num_workers=0,
        )
        for name, indices in (("train", train_indices), ("val", val_indices), ("test", test_indices))
    }
    model = ActiveReturnCritic()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    positive = max(int(np.sum(labels[train_indices])), 1)
    negative = int(train_indices.size - positive)
    positive_weight = torch.tensor(min(negative / positive, 60.0), dtype=torch.float32)
    gate_loss = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -float("inf")
    stale = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for spatial, targets, scalars, gate, regression_target, _ in loaders["train"]:
            optimizer.zero_grad()
            logits, regression = model(spatial, targets, scalars)
            loss_gate = gate_loss(logits, gate)
            loss_regression = nn.functional.smooth_l1_loss(regression, regression_target)
            loss = loss_gate + 0.25 * loss_regression
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_logits, _, val_labels, _ = predict(model, loaders["val"])
        val_auprc = float(average_precision_score(val_labels, val_logits))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auprc": val_auprc})
        if val_auprc > best_score + 1.0e-5:
            best_score = val_auprc
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= int(args.patience):
            break
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    val_logits, _, val_labels, _ = predict(model, loaders["val"])
    temperature = calibrate_temperature(val_logits, val_labels)
    test_logits, test_regression, test_labels, test_index = predict(model, loaders["test"])
    result = metrics(test_logits, test_labels, temperature)
    result.update(
        {
            "seed": seed,
            "temperature": temperature,
            "train_count": int(train_indices.size),
            "val_count": int(val_indices.size),
            "test_count": int(test_indices.size),
            "train_positive_count": int(np.sum(labels[train_indices])),
            "val_positive_count": int(np.sum(labels[val_indices])),
            "test_positive_count": int(np.sum(labels[test_indices])),
            "best_epoch": int(history[int(np.argmax([item["val_auprc"] for item in history]))]["epoch"]),
        }
    )
    seed_dir = args.out_dir / f"seed_{seed}"
    seed_dir.mkdir()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "temperature": temperature,
            "seed": seed,
            "model_label": "active_return_local_fullwave_proxy_pretraining_only",
        },
        seed_dir / "checkpoint.pt",
    )
    write_csv(seed_dir / "history.csv", history)
    probability = 1.0 / (1.0 + np.exp(-np.clip(test_logits / temperature, -40.0, 40.0)))
    predictions = [
        {
            "sample_index": int(index),
            "label": int(label),
            "probability": float(probability[position]),
            "predicted_worst_active_rl_db": float(test_regression[position, 0] * 20.0),
            "predicted_total_rl_db": float(test_regression[position, 1] * 20.0),
        }
        for position, (index, label) in enumerate(zip(test_index, test_labels))
    ]
    write_csv(seed_dir / "test_predictions.csv", predictions)
    (seed_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    raw = np.load(args.dataset, allow_pickle=False)
    arrays = {name: np.asarray(raw[name]) for name in raw.files}
    seed_metrics = [train_seed(args, arrays, int(seed)) for seed in args.seeds]
    write_csv(args.out_dir / "five_seed_metrics.csv", seed_metrics)
    metric_names = ("auroc", "auprc", "brier", "ece", "precision", "recall", "f1")
    aggregate = {
        name: {
            "mean": float(np.mean([float(item[name]) for item in seed_metrics])),
            "std": float(np.std([float(item[name]) for item in seed_metrics], ddof=1)),
            "min": float(np.min([float(item[name]) for item in seed_metrics])),
            "max": float(np.max([float(item[name]) for item in seed_metrics])),
        }
        for name in metric_names
    }
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_label": "active_return_local_fullwave_proxy_pretraining_only",
        "dataset": str(args.dataset),
        "sample_count": int(arrays["proxy_gate"].size),
        "positive_count": int(np.sum(arrays["proxy_gate"])),
        "five_seed_metrics": aggregate,
        "checkpoint_use": "candidate screening initialization only; full-wave fine-tuning and scene-level retest required",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    main()
