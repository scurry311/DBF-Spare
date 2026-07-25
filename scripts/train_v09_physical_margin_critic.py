#!/usr/bin/env python3
"""Train the v0.9 scene-conditioned physical-margin residual critic."""

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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "hfss_outputs" / "v09_margin_development_dataset_20260726_run01"
DEFAULT_OUT = ROOT / "hfss_outputs" / "v09_physical_margin_critic_20260726_run02"
KMAX = 6
MARGIN_SCALE = np.asarray([3.0, 5.0, 5.0, 0.5, 2.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--scene-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=2.0e-4)
    parser.add_argument("--rank-weight", type=float, default=0.25)
    parser.add_argument("--sign-weight", type=float, default=0.35)
    parser.add_argument("--uncertainty-kappa", type=float, default=1.0)
    parser.add_argument("--seeds", default="20260726,20260727,20260728,20260729,20260730")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
    return np.stack(
        [
            valid,
            np.rad2deg(theta) / 90.0 * valid,
            np.sin(phi) * valid,
            np.cos(phi) * valid,
            np.sin(theta) * np.cos(phi) * valid,
            np.sin(theta) * np.sin(phi) * valid,
            np.cos(theta) * valid,
        ],
        axis=-1,
    ).astype(np.float32)


def spatial_features(masks: np.ndarray, weights_ri: np.ndarray) -> np.ndarray:
    complex_weights = weights_ri[..., 0] + 1j * weights_ri[..., 1]
    norms = np.linalg.norm(complex_weights, axis=1, keepdims=True)
    normalized = complex_weights / np.maximum(norms, 1.0e-8)
    channels = np.stack((normalized.real, normalized.imag), axis=2)
    channels = channels.transpose(0, 2, 3, 1).reshape(-1, KMAX * 2, 16, 16)
    mask_channel = masks.reshape(-1, 1, 16, 16).astype(np.float32)
    return np.concatenate((mask_channel, channels.astype(np.float32)), axis=1)


def scalar_features(data: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    weights_ri = np.asarray(data["nominal_external_task_weights_real_imag"], dtype=np.float32)
    weights = weights_ri[..., 0] + 1j * weights_ri[..., 1]
    task_norms = np.linalg.norm(weights, axis=1)
    valid = np.asarray(data["task_valid"], dtype=bool)
    valid_norms = np.where(valid, task_norms, np.nan)
    amplitude = np.abs(weights)
    maximum = np.max(amplitude, axis=(1, 2))
    nonzero = np.where(amplitude > 1.0e-9, amplitude, np.nan)
    minimum = np.nanmin(nonzero, axis=(1, 2))
    combined_norm = np.linalg.norm(np.sum(weights, axis=2), axis=1)
    blocks = [
        np.asarray(data["nominal_margins"], dtype=np.float32),
        np.asarray(data["nominal_metrics"], dtype=np.float32),
        np.asarray(data["reference_metrics"], dtype=np.float32),
        np.column_stack(
            [
                np.asarray(data["k_values"], dtype=float) / 6.0,
                np.asarray(data["active_ratios_requested"], dtype=float),
                np.asarray(data["num_active"], dtype=float) / 256.0,
                np.asarray(data["min_target_separation_deg"], dtype=float) / 90.0,
                np.asarray(data["max_target_theta_deg"], dtype=float) / 90.0,
                np.asarray(data["large_scan"], dtype=float),
                np.asarray(data["phase_error_rms_deg"], dtype=float) / 20.0,
                np.asarray(data["gain_error_rms_db"], dtype=float) / 2.0,
                np.asarray(data["dropout_count"], dtype=float) / 8.0,
                np.asarray(data["phase_bits"], dtype=float) / 8.0,
                np.asarray(data["amplitude_bits"], dtype=float) / 8.0,
                np.asarray(data["implementation_delta_norm"], dtype=float),
                np.asarray(data["implementation_delta_max"], dtype=float),
                np.nanmin(valid_norms, axis=1),
                np.nanmean(valid_norms, axis=1),
                np.nanmax(valid_norms, axis=1),
                combined_norm,
                maximum,
                20.0 * np.log10(np.maximum(maximum / np.maximum(minimum, 1.0e-9), 1.0)),
            ]
        ).astype(np.float32),
    ]
    names = [f"nominal_margin_{name}" for name in data["margin_names"].tolist()]
    names += [f"nominal_{name}" for name in data["metric_names"].tolist()]
    names += [f"reference_{name}" for name in data["metric_names"].tolist()]
    names += [
        "k_scaled",
        "ratio",
        "num_active_scaled",
        "min_separation_scaled",
        "max_theta_scaled",
        "large_scan",
        "phase_error_scaled",
        "gain_error_scaled",
        "dropout_scaled",
        "phase_bits_scaled",
        "amplitude_bits_scaled",
        "implementation_delta_norm",
        "implementation_delta_max",
        "task_norm_min",
        "task_norm_mean",
        "task_norm_max",
        "combined_norm",
        "max_channel_amplitude",
        "channel_dynamic_range_db",
    ]
    values = np.concatenate(blocks, axis=1).astype(np.float32)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Non-finite scalar feature")
    return values, names


class PhysicalMarginCritic(nn.Module):
    def __init__(self, scalar_dim: int, margin_dim: int = 5):
        super().__init__()
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(13, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.GroupNorm(6, 48),
            nn.SiLU(),
            nn.Conv2d(48, 72, 3, stride=2, padding=1),
            nn.GroupNorm(8, 72),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(72, 96),
            nn.SiLU(),
        )
        self.target_item = nn.Sequential(nn.Linear(7, 48), nn.SiLU(), nn.Linear(48, 64), nn.SiLU())
        self.target_project = nn.Sequential(nn.Linear(128, 96), nn.SiLU())
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 80), nn.LayerNorm(80), nn.SiLU(), nn.Linear(80, 64), nn.SiLU()
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
        self.residual_mean = nn.Linear(160, margin_dim)
        self.residual_logvar = nn.Linear(160, margin_dim)

    def forward(
        self, spatial: torch.Tensor, targets: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_embedding = self.spatial_encoder(spatial)
        items = self.target_item(targets)
        valid = targets[..., :1]
        mean = (items * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        maximum = items.masked_fill(valid <= 0.0, -1.0e4).max(dim=1).values
        target_embedding = self.target_project(torch.cat((mean, maximum), dim=1))
        scalar_embedding = self.scalar_encoder(scalars)
        hidden = self.trunk(torch.cat((spatial_embedding, target_embedding, scalar_embedding), dim=1))
        return self.residual_mean(hidden), self.residual_logvar(hidden).clamp(-7.0, 4.0)


def heteroscedastic_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    row_weight: torch.Tensor,
) -> torch.Tensor:
    value = 0.5 * (torch.exp(-logvar) * (mean - target) ** 2 + logvar)
    value = value.mean(dim=1) * row_weight
    return value.sum() / row_weight.sum().clamp_min(1.0)


def listwise_loss(
    conservative_margin: torch.Tensor,
    actual_margin: torch.Tensor,
    ratio: torch.Tensor,
    scene: torch.Tensor,
) -> torch.Tensor:
    scale = torch.as_tensor(MARGIN_SCALE, dtype=conservative_margin.dtype, device=conservative_margin.device)
    predicted = torch.min(conservative_margin / scale, dim=1).values - 0.15 * ratio
    target = torch.min(actual_margin / scale, dim=1).values - 0.15 * ratio
    losses: list[torch.Tensor] = []
    for value in torch.unique(scene):
        group = scene == value
        if int(group.sum()) < 2:
            continue
        target_probability = torch.softmax(target[group].detach() / 0.5, dim=0)
        losses.append(-(target_probability * torch.log_softmax(predicted[group] / 0.5, dim=0)).sum())
    return torch.stack(losses).mean() if losses else predicted.sum() * 0.0


def normal_cdf(values: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(values.astype(np.float32))
    return (0.5 * (1.0 + torch.erf(tensor / math.sqrt(2.0)))).numpy()


def derived_probabilities(predicted_margin: np.ndarray, sigma: np.ndarray) -> dict[str, np.ndarray]:
    sigma = np.maximum(sigma, 0.10)
    component = normal_cdf(predicted_margin / sigma)
    gate15_component = component[:, :3].copy()
    gate15_component[:, 2] = normal_cdf((predicted_margin[:, 2] + 5.0) / sigma[:, 2])
    return {
        "gate15": np.prod(gate15_component, axis=1),
        "strict_gate20": np.prod(component, axis=1),
        "mainlobe_gate": component[:, 3],
        "active_rl_gate": component[:, 4],
    }


def calibrate_temperature(probability: np.ndarray, target: np.ndarray) -> float:
    probability = np.clip(probability, 1.0e-6, 1.0 - 1.0e-6)
    logit = np.log(probability / (1.0 - probability))
    best = (float("inf"), 1.0)
    for temperature in np.linspace(0.40, 4.0, 181):
        calibrated = 1.0 / (1.0 + np.exp(-logit / temperature))
        nll = -np.mean(
            target * np.log(np.maximum(calibrated, 1.0e-9))
            + (1.0 - target) * np.log(np.maximum(1.0 - calibrated, 1.0e-9))
        )
        if nll < best[0]:
            best = (float(nll), float(temperature))
    return best[1]


def apply_temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    probability = np.clip(probability, 1.0e-6, 1.0 - 1.0e-6)
    logit = np.log(probability / (1.0 - probability))
    return 1.0 / (1.0 + np.exp(-logit / float(temperature)))


def expected_calibration_error(probability: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        member = (probability >= low) & (probability < high if high < 1.0 else probability <= high)
        if np.any(member):
            value += float(np.mean(member)) * abs(float(np.mean(probability[member])) - float(np.mean(target[member])))
    return value


def binary_metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    target = target.astype(int)
    prediction = probability >= 0.5
    tp = int(np.sum(prediction & (target == 1)))
    fp = int(np.sum(prediction & (target == 0)))
    fn = int(np.sum(~prediction & (target == 1)))
    try:
        auroc = float(roc_auc_score(target, probability))
        auprc = float(average_precision_score(target, probability))
    except ValueError:
        auroc = float("nan")
        auprc = float("nan")
    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier": float(np.mean((probability - target) ** 2)),
        "ece": expected_calibration_error(probability, target),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "positive_rate": float(np.mean(target)),
    }


def forward_numpy(
    model: PhysicalMarginCritic,
    tensors: dict[str, torch.Tensor],
    indices: np.ndarray,
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        index = torch.as_tensor(indices, dtype=torch.long, device=tensors["spatial"].device)
        mean, logvar = model(tensors["spatial"][index], tensors["targets"][index], tensors["scalars"][index])
    mean_np = mean.cpu().numpy() * residual_std + residual_mean
    sigma_np = np.exp(0.5 * logvar.cpu().numpy()) * residual_std
    return mean_np, sigma_np


def ranking_metrics(
    indices: np.ndarray,
    scene: np.ndarray,
    score: np.ndarray,
    strict_gate: np.ndarray,
    ratios: np.ndarray,
    nominal_score: np.ndarray,
    fixed_ratio: float,
) -> dict[str, float | int]:
    selected: list[int] = []
    oracle = 0
    fixed_selected: list[int] = []
    for scene_id in np.unique(scene[indices]):
        group = indices[scene[indices] == scene_id]
        selected.append(int(group[np.argmax(score[group])]))
        oracle += int(np.any(strict_gate[group]))
        fixed = group[np.isclose(ratios[group], fixed_ratio)]
        if fixed.size:
            fixed_selected.append(int(fixed[np.argmax(nominal_score[fixed])]))
    return {
        "scene_count": len(selected),
        "top1_strict_rate": float(np.mean(strict_gate[selected])) if selected else float("nan"),
        "oracle_strict_rate": oracle / max(len(selected), 1),
        "top1_mean_ratio": float(np.mean(ratios[selected])) if selected else float("nan"),
        "fixed_strategy_rate": float(np.mean(strict_gate[fixed_selected])) if fixed_selected else float("nan"),
        "fixed_strategy_scene_count": len(fixed_selected),
    }


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite v0.9 critic run: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with np.load(args.dataset_dir / "dataset_arrays.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    if np.any(np.isclose(data["active_ratios_requested"], 1.0)):
        raise RuntimeError("Training dataset contains ratio=1.0 control")
    if any("control" in str(value).lower() for value in data["selection_roles"]):
        raise RuntimeError("Training dataset contains a nominal control role")

    split_id = np.asarray(data["split_id"], dtype=int)
    train_indices = np.flatnonzero(split_id == 0)
    val_indices = np.flatnonzero(split_id == 1)
    test_indices = np.flatnonzero(split_id == 2)
    train_scenes = set(np.asarray(data["sample_index"])[train_indices].tolist())
    val_scenes = set(np.asarray(data["sample_index"])[val_indices].tolist())
    test_scenes = set(np.asarray(data["sample_index"])[test_indices].tolist())
    if train_scenes & val_scenes or train_scenes & test_scenes or val_scenes & test_scenes:
        raise RuntimeError("Scene leakage across train/val/test")

    spatial = spatial_features(data["masks"], data["nominal_external_task_weights_real_imag"])
    targets = target_features(data["targets_deg"], data["task_valid"])
    scalars_raw, scalar_names = scalar_features(data)
    scalar_mean = scalars_raw[train_indices].mean(axis=0, keepdims=True)
    scalar_std = scalars_raw[train_indices].std(axis=0, keepdims=True)
    scalar_std = np.where(scalar_std < 1.0e-6, 1.0, scalar_std)
    scalars = ((scalars_raw - scalar_mean) / scalar_std).astype(np.float32)
    residual_raw = np.asarray(data["margin_residuals"], dtype=np.float32)
    residual_mean = residual_raw[train_indices].mean(axis=0, keepdims=True)
    residual_std = residual_raw[train_indices].std(axis=0, keepdims=True)
    residual_std = np.where(residual_std < 0.05, 0.05, residual_std)
    residual = ((residual_raw - residual_mean) / residual_std).astype(np.float32)
    nominal_margin = np.asarray(data["nominal_margins"], dtype=np.float32)
    actual_margin = np.asarray(data["actual_margins"], dtype=np.float32)
    ratios = np.asarray(data["active_ratios_requested"], dtype=np.float32)
    scene = np.asarray(data["sample_index"], dtype=np.int64)
    hard_negative = np.asarray(data["hard_negative"], dtype=np.float32)
    near_boundary = np.asarray(data["near_boundary"], dtype=np.float32)
    strict_gate = np.asarray(data["strict_gate20"], dtype=np.int8)
    strategy = np.asarray(
        [f"r{float(ratio):.1f}_{kind}" for ratio, kind in zip(ratios, data["variant_kind"])]
    )
    nominal_score = np.min(nominal_margin / MARGIN_SCALE[None, :], axis=1)
    fixed_ratio_rates: dict[float, float] = {}
    for fixed_ratio_candidate in sorted(np.unique(ratios)):
        chosen: list[int] = []
        for scene_id in sorted(train_scenes):
            group = train_indices[
                (scene[train_indices] == scene_id)
                & np.isclose(ratios[train_indices], fixed_ratio_candidate)
            ]
            chosen.append(int(group[np.argmax(nominal_score[group])]))
        fixed_ratio_rates[float(fixed_ratio_candidate)] = float(np.mean(strict_gate[chosen]))
    fixed_ratio = max(fixed_ratio_rates, key=fixed_ratio_rates.get)
    fixed_strategy = f"ratio_{fixed_ratio:.1f}_nominal_margin"
    device = resolve_device(args.device)
    tensors = {
        "spatial": torch.from_numpy(spatial).to(device),
        "targets": torch.from_numpy(targets).to(device),
        "scalars": torch.from_numpy(scalars).to(device),
        "residual": torch.from_numpy(residual).to(device),
        "nominal": torch.from_numpy(nominal_margin).to(device),
        "actual": torch.from_numpy(actual_margin).to(device),
        "ratio": torch.from_numpy(ratios).to(device),
        "scene": torch.from_numpy(scene).to(device),
        "weight": torch.from_numpy(1.0 + 2.0 * hard_negative + near_boundary).to(device),
    }
    residual_mean_t = torch.from_numpy(residual_mean.astype(np.float32)).to(device)
    residual_std_t = torch.from_numpy(residual_std.astype(np.float32)).to(device)

    seed_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    for seed in seeds:
        set_seed(seed)
        model = PhysicalMarginCritic(scalars.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_state: dict[str, torch.Tensor] | None = None
        best_val = float("inf")
        stale = 0
        rng = np.random.default_rng(seed)
        train_scene_values = np.asarray(sorted(train_scenes), dtype=np.int64)
        history: list[dict[str, Any]] = []
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            rng.shuffle(train_scene_values)
            train_losses: list[float] = []
            for start in range(0, train_scene_values.size, int(args.scene_batch_size)):
                scene_batch = train_scene_values[start : start + int(args.scene_batch_size)]
                indices_np = train_indices[np.isin(scene[train_indices], scene_batch)]
                indices = torch.as_tensor(indices_np, dtype=torch.long, device=device)
                mean_std, logvar = model(
                    tensors["spatial"][indices], tensors["targets"][indices], tensors["scalars"][indices]
                )
                regression = heteroscedastic_loss(
                    mean_std, logvar, tensors["residual"][indices], tensors["weight"][indices]
                )
                predicted_residual = mean_std * residual_std_t + residual_mean_t
                predicted_margin = tensors["nominal"][indices] + predicted_residual
                sign = torch.nn.functional.binary_cross_entropy_with_logits(
                    predicted_margin,
                    (tensors["actual"][indices] >= 0.0).float(),
                    reduction="none",
                )
                sign = (sign.mean(dim=1) * tensors["weight"][indices]).sum() / tensors["weight"][indices].sum()
                sigma = torch.exp(0.5 * logvar) * residual_std_t
                conservative = predicted_margin - float(args.uncertainty_kappa) * sigma
                rank = listwise_loss(
                    conservative,
                    tensors["actual"][indices],
                    tensors["ratio"][indices],
                    tensors["scene"][indices],
                )
                loss = regression + float(args.sign_weight) * sign + float(args.rank_weight) * rank
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            model.eval()
            with torch.no_grad():
                val = torch.as_tensor(val_indices, dtype=torch.long, device=device)
                val_mean, val_logvar = model(tensors["spatial"][val], tensors["targets"][val], tensors["scalars"][val])
                val_loss = float(
                    heteroscedastic_loss(
                        val_mean, val_logvar, tensors["residual"][val], tensors["weight"][val]
                    ).cpu()
                )
            history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_nll": val_loss})
            if val_loss < best_val - 1.0e-4:
                best_val = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= int(args.patience):
                    break
        if best_state is None:
            raise RuntimeError("No checkpoint produced")
        model.load_state_dict(best_state)
        mean_all, sigma_all = forward_numpy(model, tensors, np.arange(scene.size), residual_mean, residual_std)
        predicted_margin = nominal_margin + mean_all
        probabilities = derived_probabilities(predicted_margin, sigma_all)
        temperatures: dict[str, float] = {}
        labels = {
            "gate15": np.asarray(data["gate15"], dtype=np.int8),
            "strict_gate20": strict_gate,
            "mainlobe_gate": np.asarray(data["mainlobe_gate"], dtype=np.int8),
            "active_rl_gate": np.asarray(data["active_rl_gate"], dtype=np.int8),
        }
        for name in probabilities:
            temperatures[name] = calibrate_temperature(probabilities[name][val_indices], labels[name][val_indices])
            probabilities[name] = apply_temperature(probabilities[name], temperatures[name])
        conservative = predicted_margin - float(args.uncertainty_kappa) * sigma_all
        score = np.min(conservative / MARGIN_SCALE[None, :], axis=1) - 0.15 * ratios
        ranking = ranking_metrics(
            test_indices,
            scene,
            score,
            strict_gate,
            ratios,
            nominal_score,
            fixed_ratio,
        )
        row: dict[str, Any] = {
            "seed": seed,
            "epochs": len(history),
            "best_val_nll": best_val,
            "fixed_strategy": fixed_strategy,
            **ranking,
        }
        for name in probabilities:
            for key, value in binary_metrics(probabilities[name][test_indices], labels[name][test_indices]).items():
                row[f"test_{name}_{key}"] = value
            row[f"temperature_{name}"] = temperatures[name]
        seed_rows.append(row)
        seed_dir = args.out_dir / f"seed_{seed}"
        seed_dir.mkdir()
        torch.save(
            {
                "model_state": best_state,
                "scalar_names": scalar_names,
                "scalar_mean": scalar_mean,
                "scalar_std": scalar_std,
                "residual_mean": residual_mean,
                "residual_std": residual_std,
                "margin_names": data["margin_names"],
                "temperatures": temperatures,
                "uncertainty_kappa": float(args.uncertainty_kappa),
            },
            seed_dir / "best_checkpoint.pt",
        )
        write_csv(seed_dir / "training_history.csv", history)
        for index in test_indices:
            prediction: dict[str, Any] = {
                "seed": seed,
                "candidate_index": int(index),
                "source_candidate_index": int(data["source_candidate_indices"][index]),
                "sample_index": int(scene[index]),
                "k_value": int(data["k_values"][index]),
                "ratio": float(ratios[index]),
                "strategy": str(strategy[index]),
                "strict_gate20": int(strict_gate[index]),
                "ranking_score": float(score[index]),
            }
            for name in probabilities:
                prediction[f"prob_{name}"] = float(probabilities[name][index])
            for margin_index, name in enumerate(data["margin_names"].tolist()):
                prediction[f"pred_margin_{name}_db"] = float(predicted_margin[index, margin_index])
                prediction[f"sigma_margin_{name}_db"] = float(sigma_all[index, margin_index])
                prediction[f"actual_margin_{name}_db"] = float(actual_margin[index, margin_index])
            prediction_rows.append(prediction)
        print(json.dumps(row), flush=True)

    write_csv(args.out_dir / "five_seed_metrics.csv", seed_rows)
    write_csv(args.out_dir / "test_predictions.csv", prediction_rows)
    aggregate: dict[str, Any] = {}
    numeric_keys = [
        key for key in seed_rows[0] if key not in ("seed", "fixed_strategy")
        and isinstance(seed_rows[0][key], (int, float, np.integer, np.floating))
    ]
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in seed_rows], dtype=float)
        aggregate[key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0,
            "ci95_half_width": float(1.96 * np.nanstd(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0,
        }
    acceptance = {
        "gate15_auroc_ge_0_88": aggregate["test_gate15_auroc"]["mean"] >= 0.88,
        "strict_auroc_ge_0_88": aggregate["test_strict_gate20_auroc"]["mean"] >= 0.88,
        "gate15_ece_le_0_08": aggregate["test_gate15_ece"]["mean"] <= 0.08,
        "strict_ece_le_0_08": aggregate["test_strict_gate20_ece"]["mean"] <= 0.08,
        "top1_beats_fixed_strategy": aggregate["top1_strict_rate"]["mean"] > aggregate["fixed_strategy_rate"]["mean"],
    }
    acceptance["open_hfss_smoke"] = bool(all(acceptance.values()))
    summary = {
        "model_version": "v0.9-scene-conditioned-physical-margin-residual",
        "dataset_dir": str(args.dataset_dir.resolve()),
        "device": str(device),
        "candidate_count": int(scene.size),
        "scene_counts": {
            "train": len(train_scenes), "val": len(val_scenes), "test": len(test_scenes)
        },
        "contains_nominal_control": False,
        "v08_used_for_tuning": False,
        "fixed_strategy": fixed_strategy,
        "fixed_ratio_train_rates": fixed_ratio_rates,
        "five_seed_aggregate": aggregate,
        "acceptance": acceptance,
        "elapsed_seconds": time.time() - started,
    }
    (args.out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
