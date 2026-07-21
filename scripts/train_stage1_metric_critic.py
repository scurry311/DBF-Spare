"""Train the stage-1 HFSS-aware metric critic.

The critic predicts PSLL, nearest/local isolation, energy, and gate-pass
probability from scene features, mask, and projected complex weights. It is a
fast screening model for candidate sparse masks; it is not the final generator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "hfss_outputs" / "multitask_dataset"
DEFAULT_METRIC_DIR = DEFAULT_DATASET_DIR / "stage1_metric_dataset"
DEFAULT_OUT_ROOT = DEFAULT_DATASET_DIR / "training_runs"
K_VALUES = (1, 2, 4, 6)
KMAX = 6
NUM_ELEMENTS = 256
METRIC_NAMES = ("psll_db", "iso_nearest_db", "iso_local_db", "energy_proxy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-dir", type=Path, default=DEFAULT_METRIC_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="stage1_metric_critic_mvp")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--lr", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hfss-variant-train", action="store_true")
    parser.add_argument("--fullwave-label-weight", type=float, default=3.0)
    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=1.0,
        help=(
            "Extra multiplicative weight for full-wave hard negatives where "
            "AF gate passes but HFSS full-wave gate fails. Keep 1.0 to disable."
        ),
    )
    parser.add_argument("--gate-loss-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    arrays = np.load(path, allow_pickle=False)
    return {key: arrays[key] for key in arrays.files}


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


def build_features(data: dict[str, np.ndarray]) -> np.ndarray:
    n = data["sample_index"].shape[0]
    k_values = data["k_values"].astype(np.int64)
    active_ratios = data["active_ratios_requested"].astype(np.float32)
    num_active = data["num_active"].astype(np.float32)
    targets_deg = np.nan_to_num(data["targets_deg"].astype(np.float32), nan=0.0)
    task_valid = data["task_valid"].astype(np.float32)
    masks = data["masks"].astype(np.float32)
    weights = data["weights_real_imag"].astype(np.float32) * num_active[:, None, None, None].clip(min=1.0)

    k_onehot = np.zeros((n, len(K_VALUES)), dtype=np.float32)
    for col, kval in enumerate(K_VALUES):
        k_onehot[:, col] = (k_values == kval).astype(np.float32)
    dirs = unit_vectors(targets_deg[:, :, 0], targets_deg[:, :, 1])
    dirs = np.where(task_valid[:, :, None] > 0.0, dirs, 0.0)
    target_features = np.concatenate(
        [
            task_valid[:, :, None],
            (targets_deg[:, :, :1] / 90.0) * task_valid[:, :, None],
            np.sin(np.deg2rad(targets_deg[:, :, 1:2])) * task_valid[:, :, None],
            np.cos(np.deg2rad(targets_deg[:, :, 1:2])) * task_valid[:, :, None],
            dirs,
        ],
        axis=-1,
    ).reshape(n, -1)
    weight_abs = np.sqrt(weights[..., 0] ** 2 + weights[..., 1] ** 2)
    per_task_weight_energy = np.sum(weight_abs ** 2, axis=1)
    global_stats = np.stack(
        [
            k_values.astype(np.float32) / float(KMAX),
            active_ratios,
            num_active / float(NUM_ELEMENTS),
            masks.mean(axis=1),
            masks.std(axis=1),
            weight_abs.mean(axis=(1, 2)),
            weight_abs.max(axis=(1, 2)),
            weight_abs.std(axis=(1, 2)),
        ],
        axis=1,
    )
    return np.concatenate(
        [
            k_onehot,
            global_stats,
            target_features,
            per_task_weight_energy.astype(np.float32),
            masks,
            weights.reshape(n, -1),
        ],
        axis=1,
    ).astype(np.float32)


def build_targets(data: dict[str, np.ndarray], *, fullwave_label_weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    label_names = [str(x) for x in data["label_names"]]
    labels = data["labels"].astype(np.float32)
    col = {name: i for i, name in enumerate(label_names)}

    y = np.full((labels.shape[0], len(METRIC_NAMES)), np.nan, dtype=np.float32)
    weights = np.zeros_like(y, dtype=np.float32)
    # PSLL has full-wave labels for all original samples and task-HFSS variants.
    for out_col, (hfss_name, af_name) in enumerate(
        [
            ("hfss_psll_db", "af_psll_db"),
            ("hfss_iso_nearest_db", "af_iso_nearest_db"),
            ("hfss_iso_local_db", "af_iso_local_db"),
            ("energy_proxy", "energy_proxy"),
        ]
    ):
        hfss = labels[:, col[hfss_name]] if hfss_name in col else np.full(labels.shape[0], np.nan, dtype=np.float32)
        af = labels[:, col[af_name]]
        use_hfss = np.isfinite(hfss)
        y[:, out_col] = np.where(use_hfss, hfss, af)
        weights[:, out_col] = np.where(
            np.isfinite(y[:, out_col]),
            np.where(use_hfss, float(fullwave_label_weight), 1.0),
            0.0,
        )
    gate_hfss = labels[:, col["hfss_gate_pass"]]
    gate_af = labels[:, col["af_gate_pass"]]
    use_hfss_gate = np.isfinite(gate_hfss)
    gate = np.where(use_hfss_gate, gate_hfss, gate_af).astype(np.float32)
    gate_weight = np.where(np.isfinite(gate), np.where(use_hfss_gate, float(fullwave_label_weight), 1.0), 0.0).astype(np.float32)
    gate = np.nan_to_num(gate, nan=0.0)
    return y, weights, np.stack([gate, gate_weight], axis=1), list(METRIC_NAMES)


def fullwave_hard_negative_mask(data: dict[str, np.ndarray]) -> np.ndarray:
    label_names = [str(x) for x in data["label_names"]]
    labels = data["labels"].astype(np.float32)
    col = {name: i for i, name in enumerate(label_names)}
    if "af_gate_pass" not in col or "hfss_gate_pass" not in col:
        return np.zeros(labels.shape[0], dtype=np.bool_)
    af_gate = labels[:, col["af_gate_pass"]]
    hfss_gate = labels[:, col["hfss_gate_pass"]]
    return np.isfinite(af_gate) & np.isfinite(hfss_gate) & (af_gate >= 0.5) & (hfss_gate < 0.5)


def apply_hard_negative_weight(
    metric_weights: np.ndarray,
    gate: np.ndarray,
    hard_negative: np.ndarray,
    *,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    if weight <= 1.0 or not np.any(hard_negative):
        return metric_weights, gate
    metric_weights = metric_weights.copy()
    gate = gate.copy()
    multiplier = np.where(hard_negative, float(weight), 1.0).astype(np.float32)
    metric_weights *= multiplier[:, None]
    gate[:, 1] *= multiplier
    return metric_weights, gate


class MetricDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        metrics: np.ndarray,
        metric_weights: np.ndarray,
        gate: np.ndarray,
        indices: np.ndarray,
    ):
        self.features = features.astype(np.float32)
        self.metrics = metrics.astype(np.float32)
        self.metric_weights = metric_weights.astype(np.float32)
        self.gate = gate.astype(np.float32)
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        return {
            "x": torch.from_numpy(self.features[idx]),
            "y": torch.from_numpy(self.metrics[idx]),
            "w": torch.from_numpy(self.metric_weights[idx]),
            "gate": torch.tensor(self.gate[idx, 0], dtype=torch.float32),
            "gate_weight": torch.tensor(self.gate[idx, 1], dtype=torch.float32),
        }


class MetricCritic(nn.Module):
    def __init__(self, input_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
        )
        self.metric_head = nn.Linear(hidden // 2, out_dim)
        self.gate_head = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.metric_head(h), self.gate_head(h).squeeze(-1)


def standardize_train(features: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features[train_idx].mean(axis=0, keepdims=True)
    std = features[train_idx].std(axis=0, keepdims=True)
    std = np.where(std < 1.0e-6, 1.0, std)
    return ((features - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def standardize_targets(
    y: np.ndarray,
    w: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_std = y.copy()
    mean = np.zeros((1, y.shape[1]), dtype=np.float32)
    std = np.ones((1, y.shape[1]), dtype=np.float32)
    for col in range(y.shape[1]):
        mask = (w[train_idx, col] > 0.0) & np.isfinite(y[train_idx, col])
        vals = y[train_idx[mask], col]
        if vals.size:
            mean[0, col] = float(vals.mean())
            std[0, col] = float(vals.std()) if vals.std() > 1.0e-6 else 1.0
            y_std[:, col] = (y[:, col] - mean[0, col]) / std[0, col]
    return y_std.astype(np.float32), mean, std


def split_indices(data: dict[str, np.ndarray], include_fullwave_variants: bool) -> dict[str, np.ndarray]:
    source = data["source"].astype(str)
    split_id = data["split_id"].astype(np.int64)
    is_original = source == "original"
    out: dict[str, np.ndarray] = {
        "train": np.flatnonzero((split_id == 0) & is_original),
        "val": np.flatnonzero((split_id == 1) & is_original),
        "test": np.flatnonzero((split_id == 2) & is_original),
        "hfss_variants": np.flatnonzero(~is_original),
    }
    if include_fullwave_variants:
        out["train"] = np.concatenate([out["train"], out["hfss_variants"]])
    return out


def masked_huber(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    valid = weight > 0.0
    if not torch.any(valid):
        return pred.sum() * 0.0
    loss = torch.nn.functional.smooth_l1_loss(pred[valid], target[valid], reduction="none")
    weighted = loss * weight[valid]
    return weighted.sum() / weight[valid].sum().clamp_min(1.0)


def weighted_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    valid = weight > 0.0
    if not torch.any(valid):
        return logits.sum() * 0.0
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[valid], target[valid], reduction="none")
    weighted = loss * weight[valid]
    return weighted.sum() / weight[valid].sum().clamp_min(1.0)


def finite_mae(pred: np.ndarray, target: np.ndarray, weight: np.ndarray) -> float:
    mask = (weight > 0.0) & np.isfinite(target) & np.isfinite(pred)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - target[mask])))


def finite_rmse(pred: np.ndarray, target: np.ndarray, weight: np.ndarray) -> float:
    mask = (weight > 0.0) & np.isfinite(target) & np.isfinite(pred)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - target[mask]) ** 2)))


def gate_stats(logits: np.ndarray, gate: np.ndarray, gate_weight: np.ndarray) -> dict[str, float]:
    mask = gate_weight > 0.0
    if not np.any(mask):
        return {"gate_accuracy": float("nan"), "gate_positive_rate": float("nan")}
    prob = 1.0 / (1.0 + np.exp(-logits[mask]))
    pred = (prob >= 0.5).astype(np.float32)
    return {
        "gate_accuracy": float(np.mean(pred == gate[mask])),
        "gate_positive_rate": float(np.mean(gate[mask])),
        "gate_prob_mean": float(np.mean(prob)),
    }


def predict_all(
    model: MetricCritic,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            x = torch.from_numpy(features[start : start + batch_size]).to(device)
            metric_std, gate_logit = model(x)
            metric = metric_std.cpu().numpy() * y_std + y_mean
            preds.append(metric.astype(np.float32))
            gates.append(gate_logit.cpu().numpy().astype(np.float32))
    return np.concatenate(preds, axis=0), np.concatenate(gates, axis=0)


def summarize_split(
    name: str,
    indices: np.ndarray,
    pred: np.ndarray,
    gate_logits: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    gate: np.ndarray,
    metric_names: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {"split": name, "n": int(indices.shape[0])}
    for col, metric_name in enumerate(metric_names):
        row[f"{metric_name}_mae"] = finite_mae(pred[indices, col], y[indices, col], w[indices, col])
        row[f"{metric_name}_rmse"] = finite_rmse(pred[indices, col], y[indices, col], w[indices, col])
    row.update(gate_stats(gate_logits[indices], gate[indices, 0], gate[indices, 1]))
    return row


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


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


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    out_dir = args.out_dir or DEFAULT_OUT_ROOT / str(args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_npz(args.metric_dir / "stage1_metric_dataset.npz")
    features = build_features(data)
    y, metric_weights, gate, metric_names = build_targets(data, fullwave_label_weight=float(args.fullwave_label_weight))
    hard_negative = fullwave_hard_negative_mask(data)
    metric_weights, gate = apply_hard_negative_weight(
        metric_weights,
        gate,
        hard_negative,
        weight=float(args.hard_negative_weight),
    )
    split = split_indices(data, include_fullwave_variants=bool(args.hfss_variant_train))
    features_std, x_mean, x_std = standardize_train(features, split["train"])
    y_std, y_mean, y_scale = standardize_targets(y, metric_weights, split["train"])

    train_ds = MetricDataset(features_std, y_std, metric_weights, gate, split["train"])
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=False)
    model = MetricCritic(features.shape[1], int(args.hidden), len(metric_names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    log_rows: list[dict[str, Any]] = []
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    start_time = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            target = batch["y"].to(device)
            weight = batch["w"].to(device)
            gate_target = batch["gate"].to(device)
            gate_weight = batch["gate_weight"].to(device)
            pred, gate_logit = model(x)
            loss_metric = masked_huber(pred, target, weight)
            loss_gate = weighted_bce(gate_logit, gate_target, gate_weight)
            loss = loss_metric + float(args.gate_loss_weight) * loss_gate
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu().item()) * x.shape[0]
            count += int(x.shape[0])

        pred_all, gate_logits = predict_all(model, features_std, int(args.batch_size), device, y_mean, y_scale)
        train_summary = summarize_split("train", split["train"], pred_all, gate_logits, y, metric_weights, gate, metric_names)
        val_summary = summarize_split("val", split["val"], pred_all, gate_logits, y, metric_weights, gate, metric_names)
        val_score = float(val_summary["psll_db_mae"]) + 0.2 * float(val_summary["energy_proxy_mae"])
        if np.isfinite(val_score) and val_score < best_val:
            best_val = val_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        log_row = {
            "epoch": epoch,
            "loss": total / max(count, 1),
            "elapsed_s": time.time() - start_time,
            "train_psll_mae": train_summary["psll_db_mae"],
            "val_psll_mae": val_summary["psll_db_mae"],
            "val_iso_nearest_mae": val_summary["iso_nearest_db_mae"],
            "val_iso_local_mae": val_summary["iso_local_db_mae"],
            "val_gate_accuracy": val_summary["gate_accuracy"],
        }
        log_rows.append(log_row)
        if epoch == 1 or epoch % 10 == 0 or epoch == int(args.epochs):
            print(json.dumps(log_row, indent=2))

    if best_state is not None:
        model.load_state_dict(best_state)
    pred_all, gate_logits = predict_all(model, features_std, int(args.batch_size), device, y_mean, y_scale)
    summaries = [
        summarize_split("train", split["train"], pred_all, gate_logits, y, metric_weights, gate, metric_names),
        summarize_split("val", split["val"], pred_all, gate_logits, y, metric_weights, gate, metric_names),
        summarize_split("test", split["test"], pred_all, gate_logits, y, metric_weights, gate, metric_names),
    ]
    if split["hfss_variants"].size:
        summaries.append(
            summarize_split(
                "hfss_variants",
                split["hfss_variants"],
                pred_all,
                gate_logits,
                y,
                metric_weights,
                gate,
                metric_names,
            )
        )
    hard_negative_indices = np.flatnonzero(hard_negative)
    if hard_negative_indices.size:
        summaries.append(
            summarize_split(
                "hard_negatives",
                hard_negative_indices,
                pred_all,
                gate_logits,
                y,
                metric_weights,
                gate,
                metric_names,
            )
        )
    high_risk = np.flatnonzero(
        (data["k_values"].astype(np.int64) == 6)
        & (data["active_ratios_requested"].astype(np.float32) <= 0.7)
    )
    summaries.append(summarize_split("k6_ratio_le_0.7", high_risk, pred_all, gate_logits, y, metric_weights, gate, metric_names))

    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": int(features.shape[1]),
            "hidden": int(args.hidden),
            "metric_names": metric_names,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_scale,
            "args": vars(args),
        },
        out_dir / "metric_critic.pt",
    )
    write_csv(out_dir / "train_log.csv", log_rows)
    write_csv(out_dir / "eval_summary.csv", summaries)
    np.savez_compressed(
        out_dir / "metric_critic_predictions.npz",
        pred_metrics=pred_all,
        gate_logits=gate_logits,
        metric_names=np.asarray(metric_names),
        sample_index=data["sample_index"],
        source=data["source"],
        hard_negative=hard_negative.astype(np.int8),
    )
    run_summary = {
        "run_name": str(args.run_name),
        "metric_dir": str(args.metric_dir),
        "out_dir": str(out_dir),
        "device": str(device),
        "epochs": int(args.epochs),
        "train_count": int(split["train"].shape[0]),
        "val_count": int(split["val"].shape[0]),
        "test_count": int(split["test"].shape[0]),
        "hfss_variant_count": int(split["hfss_variants"].shape[0]),
        "hard_negative_count": int(hard_negative.sum()),
        "hard_negative_weight": float(args.hard_negative_weight),
        "best_val_score": best_val,
        "elapsed_s": time.time() - start_time,
        "headline": {row["split"]: row for row in summaries},
        "outputs": {
            "model": str(out_dir / "metric_critic.pt"),
            "train_log": str(out_dir / "train_log.csv"),
            "eval_summary": str(out_dir / "eval_summary.csv"),
            "predictions": str(out_dir / "metric_critic_predictions.npz"),
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
