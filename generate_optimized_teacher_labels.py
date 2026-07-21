"""Generate optimized teacher labels for the HFSS multitask dataset.

Version v1 focuses on a fast, deterministic array-factor teacher:

* fixed active count from each sample's active_ratio;
* deterministic uniform-aperture initial mask;
* target-aware greedy element swaps to reduce PSLL to the weakest target;
* analytic per-task steering weights for the optimized mask;
* optional HDF5 label export under /labels/<teacher_name>.

This is intended as a teacher-label generator before a neural model learns the
optimization strategy. It does not run HFSS; HFSS can be used later to validate
the selected teacher samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - v1 can still run without torch.
    torch = None


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT / "hfss_outputs" / "multitask_dataset"
KMAX = 6
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--teacher-name", default="greedy_psll_v1")
    parser.add_argument("--max-swaps", type=int, default=8)
    parser.add_argument("--top-candidates", type=int, default=14)
    parser.add_argument("--random-pairs", type=int, default=48)
    parser.add_argument("--theta-step", type=float, default=5.0)
    parser.add_argument("--phi-step", type=float, default=10.0)
    parser.add_argument("--eval-theta-step", type=float, default=2.0)
    parser.add_argument("--eval-phi-step", type=float, default=5.0)
    parser.add_argument("--spread-margin-db", type=float, default=5.0)
    parser.add_argument("--spread-weight", type=float, default=0.20)
    parser.add_argument("--weak-peak-weight", type=float, default=0.03)
    parser.add_argument("--min-improve-db", type=float, default=0.02)
    parser.add_argument("--init-mode", choices=("deterministic", "original"), default="deterministic")
    parser.add_argument("--strategy", choices=("v1", "v2"), default="v1")
    parser.add_argument("--v2-starts", type=int, default=8)
    parser.add_argument("--v2-greedy-swaps", type=int, default=6)
    parser.add_argument("--v2-final-candidates", type=int, default=8)
    parser.add_argument("--nsga-population", type=int, default=18)
    parser.add_argument("--nsga-generations", type=int, default=4)
    parser.add_argument("--nsga-offspring", type=int, default=18)
    parser.add_argument("--softmask-steps", type=int, default=28)
    parser.add_argument("--softmask-lr", type=float, default=0.75)
    parser.add_argument("--softmask-min-k", type=int, default=4)
    parser.add_argument("--softmask-temperature-db", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--limit", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def make_grid(theta_step: float, phi_step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_values = np.arange(0.0, 90.0 + 0.1, theta_step, dtype=np.float32)
    phi_values = np.arange(0.0, 360.0, phi_step, dtype=np.float32)
    theta, phi = np.meshgrid(theta_values, phi_values, indexing="ij")
    theta_flat = theta.reshape(-1)
    phi_flat = phi.reshape(-1)
    return theta_flat, phi_flat, unit_vectors(theta_flat, phi_flat)


def deterministic_element_order(element_ixiy: np.ndarray) -> np.ndarray:
    coords = element_ixiy.astype(np.int64)
    center = (coords.max(axis=0) + coords.min(axis=0)) / 2.0
    # Deterministic aperture-spreading order: alternate far aperture and
    # low-discrepancy Morton-like ordering so every prefix remains distributed.
    ix = coords[:, 0]
    iy = coords[:, 1]
    morton = np.zeros(coords.shape[0], dtype=np.int64)
    for bit in range(4):
        morton |= ((ix >> bit) & 1) << (2 * bit)
        morton |= ((iy >> bit) & 1) << (2 * bit + 1)
    radius = np.linalg.norm(coords - center[None, :], axis=1)
    score = -0.65 * radius + 0.35 * (morton / max(float(morton.max()), 1.0))
    return np.argsort(score, kind="stable")


def initial_mask(order: np.ndarray, num_active: int, num_elements: int) -> np.ndarray:
    mask = np.zeros(num_elements, dtype=np.bool_)
    mask[order[:num_active]] = True
    return mask


def mask_from_scores(scores: np.ndarray, num_active: int) -> np.ndarray:
    mask = np.zeros(scores.shape[0], dtype=np.bool_)
    if num_active <= 0:
        return mask
    if num_active >= scores.shape[0]:
        mask[:] = True
        return mask
    chosen = np.argpartition(scores, -num_active)[-num_active:]
    mask[chosen] = True
    return mask


def normalize_mask_cardinality(mask: np.ndarray, num_active: int, rng: np.random.Generator) -> np.ndarray:
    out = mask.astype(bool).copy()
    active = np.flatnonzero(out)
    if active.size > num_active:
        drop = rng.choice(active, size=active.size - num_active, replace=False)
        out[drop] = False
    elif active.size < num_active:
        inactive = np.flatnonzero(~out)
        add = rng.choice(inactive, size=num_active - active.size, replace=False)
        out[add] = True
    return out


def unique_masks(masks: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    for mask in masks:
        key = np.packbits(mask.astype(np.uint8)).tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(mask.astype(bool).copy())
    return out


def stable_sample_seed(args: argparse.Namespace, targets_deg: np.ndarray, task_valid: np.ndarray, num_active: int) -> int:
    payload = b"|".join(
        [
            np.nan_to_num(targets_deg, nan=-999.0).round(3).astype(np.float32).tobytes(),
            task_valid.astype(np.uint8).tobytes(),
            np.asarray([num_active, args.seed], dtype=np.int64).tobytes(),
        ]
    )
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & 0xFFFFFFFF


def canonical_seed_masks(
    *,
    element_ixiy: np.ndarray,
    positions: np.ndarray,
    target_dirs: np.ndarray,
    target_contrib: np.ndarray,
    initial_order: np.ndarray,
    num_active: int,
    max_starts: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    num_elements = positions.shape[0]
    coords = element_ixiy.astype(np.float32)
    center = (coords.max(axis=0) + coords.min(axis=0)) / 2.0
    radius = np.linalg.norm(coords - center[None, :], axis=1)
    ix = coords[:, 0]
    iy = coords[:, 1]
    phase = 2.0 * np.pi * (target_dirs @ positions.T)
    coherent_sum = np.abs(np.exp(1j * phase).sum(axis=0))
    target_resp_score = np.abs(target_contrib).mean(axis=1)
    checker = ((ix + iy) % 2.0) - 0.01 * radius
    anti_checker = (1.0 - ((ix + iy) % 2.0)) - 0.01 * radius
    hash_score = np.mod(ix * 73.0 + iy * 151.0, 997.0) / 997.0
    diagonal = -np.abs(ix - iy) - 0.02 * radius

    masks = [
        initial_mask(initial_order, num_active, num_elements),
        initial_mask(np.argsort(radius, kind="stable"), num_active, num_elements),
        initial_mask(np.argsort(-radius, kind="stable"), num_active, num_elements),
        mask_from_scores(checker, num_active),
        mask_from_scores(anti_checker, num_active),
        mask_from_scores(coherent_sum - 0.03 * radius, num_active),
        mask_from_scores(target_resp_score - 0.02 * radius, num_active),
        mask_from_scores(hash_score - 0.015 * radius, num_active),
        mask_from_scores(diagonal, num_active),
    ]
    return unique_masks(masks)[:max(max_starts, 1)]


def target_dirs_for_sample(targets_deg: np.ndarray, task_valid: np.ndarray) -> np.ndarray:
    valid_targets = np.nan_to_num(targets_deg[task_valid.astype(bool)], nan=0.0)
    return unit_vectors(valid_targets[:, 0], valid_targets[:, 1])


def build_contributions(
    positions: np.ndarray,
    target_dirs: np.ndarray,
    grid_dirs: np.ndarray,
    num_active: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_phase = 2.0 * np.pi * (grid_dirs @ positions.T)
    grid_steer = np.exp(-1j * grid_phase).astype(np.complex64)  # G,N
    target_phase = 2.0 * np.pi * (target_dirs @ positions.T)
    target_tx = np.exp(1j * target_phase).astype(np.complex64).T  # N,K
    task_sum = target_tx.sum(axis=1)
    grid_contrib = (grid_steer * task_sum[None, :]).T / float(num_active)  # N,G
    target_rx = np.exp(-1j * target_phase).astype(np.complex64)  # K,N
    target_contrib = (target_rx.T[:, :, None] * target_tx[:, None, :]).sum(axis=2) / float(num_active)  # N,K
    return grid_contrib, target_contrib, target_tx


def side_mask_for_targets(grid_dirs: np.ndarray, target_dirs: np.ndarray, radius_deg: float = 8.0) -> np.ndarray:
    dots = np.clip(grid_dirs @ target_dirs.T, -1.0, 1.0)
    dists = np.rad2deg(np.arccos(dots))
    return dists.min(axis=1) > radius_deg


def objective_from_fields(
    af_grid: np.ndarray,
    target_resp: np.ndarray,
    side_mask: np.ndarray,
    *,
    spread_margin_db: float,
    spread_weight: float,
    weak_peak_weight: float,
) -> tuple[float, dict[str, float]]:
    grid_db = 10.0 * np.log10(np.maximum(np.abs(af_grid) ** 2, EPS))
    target_db = 10.0 * np.log10(np.maximum(np.abs(target_resp) ** 2, EPS))
    weak = float(target_db.min())
    strong = float(target_db.max())
    spread = strong - weak
    side = float(grid_db[side_mask].max())
    psll_weak = side - weak
    penalty = spread_weight * max(spread - spread_margin_db, 0.0) ** 2
    objective = psll_weak + penalty - weak_peak_weight * weak
    return float(objective), {
        "weak_peak_db": weak,
        "target_peak_mean_db": float(target_db.mean()),
        "target_spread_db": float(spread),
        "worst_sidelobe_db": side,
        "psll_to_weakest_peak_db": float(psll_weak),
        "objective": float(objective),
    }


def candidate_pairs(
    mask: np.ndarray,
    grid_contrib: np.ndarray,
    af_grid: np.ndarray,
    side_mask: np.ndarray,
    *,
    top_candidates: int,
    random_pairs: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    active = np.flatnonzero(mask)
    inactive = np.flatnonzero(~mask)
    if active.size == 0 or inactive.size == 0:
        return []
    grid_db = 10.0 * np.log10(np.maximum(np.abs(af_grid) ** 2, EPS))
    side_indices = np.flatnonzero(side_mask)
    peak_idx = int(side_indices[np.argmax(grid_db[side_mask])])
    peak = af_grid[peak_idx]
    c = grid_contrib[:, peak_idx]
    remove_score = np.real(c[active] * np.conj(peak))
    add_score = -np.real(c[inactive] * np.conj(peak))
    n_top_a = min(top_candidates, active.size)
    n_top_i = min(top_candidates, inactive.size)
    top_active = active[np.argpartition(remove_score, -n_top_a)[-n_top_a:]]
    top_inactive = inactive[np.argpartition(add_score, -n_top_i)[-n_top_i:]]
    pairs = [(int(a), int(b)) for a in top_active for b in top_inactive]
    if random_pairs > 0:
        rand_active = rng.choice(active, size=min(random_pairs, active.size), replace=active.size < random_pairs)
        rand_inactive = rng.choice(inactive, size=min(random_pairs, inactive.size), replace=inactive.size < random_pairs)
        pairs.extend((int(a), int(b)) for a, b in zip(rand_active, rand_inactive))
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(pairs))


def evaluate_pairs(
    pairs: list[tuple[int, int]],
    af_grid: np.ndarray,
    target_resp: np.ndarray,
    grid_contrib: np.ndarray,
    target_contrib: np.ndarray,
    side_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, tuple[int, int] | None, dict[str, float] | None]:
    if not pairs:
        return float("inf"), None, None
    off = np.asarray([p[0] for p in pairs], dtype=np.int64)
    on = np.asarray([p[1] for p in pairs], dtype=np.int64)
    delta_grid = -grid_contrib[off] + grid_contrib[on]
    delta_target = -target_contrib[off] + target_contrib[on]
    cand_grid = af_grid[None, :] + delta_grid
    cand_target = target_resp[None, :] + delta_target

    grid_db = 10.0 * np.log10(np.maximum(np.abs(cand_grid) ** 2, EPS))
    target_db = 10.0 * np.log10(np.maximum(np.abs(cand_target) ** 2, EPS))
    weak = target_db.min(axis=1)
    strong = target_db.max(axis=1)
    spread = strong - weak
    side = grid_db[:, side_mask].max(axis=1)
    psll_weak = side - weak
    score = (
        psll_weak
        + args.spread_weight * np.maximum(spread - args.spread_margin_db, 0.0) ** 2
        - args.weak_peak_weight * weak
    )
    best_idx = int(np.argmin(score))
    metrics = {
        "weak_peak_db": float(weak[best_idx]),
        "target_peak_mean_db": float(target_db[best_idx].mean()),
        "target_spread_db": float(spread[best_idx]),
        "worst_sidelobe_db": float(side[best_idx]),
        "psll_to_weakest_peak_db": float(psll_weak[best_idx]),
        "objective": float(score[best_idx]),
    }
    return float(score[best_idx]), pairs[best_idx], metrics


def population_objectives(
    population: list[np.ndarray],
    grid_contrib: np.ndarray,
    target_contrib: np.ndarray,
    side_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    if not population:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64), []
    gates = np.asarray(population, dtype=np.float32)
    cand_grid = gates @ grid_contrib
    cand_target = gates @ target_contrib
    grid_db = 10.0 * np.log10(np.maximum(np.abs(cand_grid) ** 2, EPS))
    target_db = 10.0 * np.log10(np.maximum(np.abs(cand_target) ** 2, EPS))
    weak = target_db.min(axis=1)
    strong = target_db.max(axis=1)
    spread = strong - weak
    side = grid_db[:, side_mask].max(axis=1)
    psll_weak = side - weak
    scalar = (
        psll_weak
        + args.spread_weight * np.maximum(spread - args.spread_margin_db, 0.0) ** 2
        - args.weak_peak_weight * weak
    )
    objectives = np.stack([psll_weak, spread, -weak], axis=1).astype(np.float64)
    metrics = [
        {
            "weak_peak_db": float(weak[i]),
            "target_peak_mean_db": float(target_db[i].mean()),
            "target_spread_db": float(spread[i]),
            "worst_sidelobe_db": float(side[i]),
            "psll_to_weakest_peak_db": float(psll_weak[i]),
            "objective": float(scalar[i]),
        }
        for i in range(len(population))
    ]
    return objectives, scalar.astype(np.float64), metrics


def nondominated_sort(objectives: np.ndarray) -> list[list[int]]:
    n = objectives.shape[0]
    dominates_list: list[list[int]] = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=np.int32)
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            p_dominates = np.all(objectives[p] <= objectives[q]) and np.any(objectives[p] < objectives[q])
            q_dominates = np.all(objectives[q] <= objectives[p]) and np.any(objectives[q] < objectives[p])
            if p_dominates:
                dominates_list[p].append(q)
            elif q_dominates:
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)
    front_index = 0
    while front_index < len(fronts) and fronts[front_index]:
        next_front: list[int] = []
        for p in fronts[front_index]:
            for q in dominates_list[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        front_index += 1
    return fronts


def crowding_distance(objectives: np.ndarray, front: list[int]) -> np.ndarray:
    distances = np.zeros(len(front), dtype=np.float64)
    if len(front) <= 2:
        distances[:] = np.inf
        return distances
    values = objectives[np.asarray(front)]
    for col in range(values.shape[1]):
        order = np.argsort(values[:, col], kind="stable")
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        lo = values[order[0], col]
        hi = values[order[-1], col]
        denom = max(float(hi - lo), 1.0e-9)
        for j in range(1, len(order) - 1):
            distances[order[j]] += float(values[order[j + 1], col] - values[order[j - 1], col]) / denom
    return distances


def nsga_select(
    population: list[np.ndarray],
    objectives: np.ndarray,
    scalar: np.ndarray,
    target_size: int,
) -> list[np.ndarray]:
    selected: list[int] = []
    for front in nondominated_sort(objectives):
        if len(selected) + len(front) <= target_size:
            selected.extend(front)
            continue
        distances = crowding_distance(objectives, front)
        front_arr = np.asarray(front, dtype=np.int64)
        # Prefer diverse nondominated points; scalar objective breaks ties.
        order = np.lexsort((scalar[front_arr], -distances))
        selected.extend(front_arr[order[: target_size - len(selected)]].tolist())
        break
    return unique_masks([population[i] for i in selected])


def crossover_fixed_count(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    num_active: int,
    rng: np.random.Generator,
) -> np.ndarray:
    child = parent_a & parent_b
    need = num_active - int(child.sum())
    if need > 0:
        pool = np.flatnonzero(np.logical_xor(parent_a, parent_b))
        if pool.size > 0:
            add = rng.choice(pool, size=min(need, pool.size), replace=False)
            child[add] = True
    return normalize_mask_cardinality(child, num_active, rng)


def mutate_mask(mask: np.ndarray, num_swaps: int, rng: np.random.Generator) -> np.ndarray:
    out = mask.astype(bool).copy()
    active = np.flatnonzero(out)
    inactive = np.flatnonzero(~out)
    if active.size == 0 or inactive.size == 0:
        return out
    swaps = min(num_swaps, active.size, inactive.size)
    off = rng.choice(active, size=swaps, replace=False)
    on = rng.choice(inactive, size=swaps, replace=False)
    out[off] = False
    out[on] = True
    return out


def nsga_candidates(
    seed_masks: list[np.ndarray],
    grid_contrib: np.ndarray,
    target_contrib: np.ndarray,
    side_mask: np.ndarray,
    num_active: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if args.nsga_population <= 0 or args.nsga_generations <= 0:
        return []
    population = unique_masks(seed_masks)
    while len(population) < args.nsga_population:
        parent = population[int(rng.integers(0, len(population)))]
        population.append(mutate_mask(parent, int(rng.integers(1, 4)), rng))
        population = unique_masks(population)
    population = population[: args.nsga_population]

    for _ in range(args.nsga_generations):
        offspring: list[np.ndarray] = []
        _, scalar, _ = population_objectives(population, grid_contrib, target_contrib, side_mask, args)
        parent_prob = np.exp(-(scalar - scalar.min()) / max(float(np.std(scalar)), 1.0))
        parent_prob = parent_prob / parent_prob.sum()
        for _j in range(args.nsga_offspring):
            ia, ib = rng.choice(len(population), size=2, replace=True, p=parent_prob)
            child = crossover_fixed_count(population[int(ia)], population[int(ib)], num_active, rng)
            child = mutate_mask(child, int(rng.integers(1, 4)), rng)
            offspring.append(child)
        combined = unique_masks(population + offspring)
        objectives, scalar, _ = population_objectives(combined, grid_contrib, target_contrib, side_mask, args)
        population = nsga_select(combined, objectives, scalar, args.nsga_population)

    _, scalar, _ = population_objectives(population, grid_contrib, target_contrib, side_mask, args)
    order = np.argsort(scalar, kind="stable")
    return [population[int(i)] for i in order[: max(args.v2_final_candidates, 1)]]


def softmask_gradient_candidate(
    start_mask: np.ndarray,
    grid_contrib: np.ndarray,
    target_contrib: np.ndarray,
    side_mask: np.ndarray,
    num_active: int,
    args: argparse.Namespace,
) -> np.ndarray | None:
    if torch is None or args.softmask_steps <= 0:
        return None
    logits_init = np.where(start_mask, 3.0, -3.0).astype(np.float32)
    logits = torch.tensor(logits_init, dtype=torch.float32, requires_grad=True)
    grid_contrib_t = torch.from_numpy(grid_contrib.astype(np.complex64))
    target_contrib_t = torch.from_numpy(target_contrib.astype(np.complex64))
    side_mask_t = torch.from_numpy(side_mask.astype(bool))
    optimizer = torch.optim.Adam([logits], lr=args.softmask_lr)
    temp = max(float(args.softmask_temperature_db), 0.25)
    for _ in range(args.softmask_steps):
        soft = torch.sigmoid(logits)
        gate = soft * (float(num_active) / soft.sum().clamp_min(1.0e-6))
        gate = gate.clamp(0.0, 1.0)
        gate_c = gate.to(torch.complex64)
        af_grid = torch.matmul(gate_c, grid_contrib_t)
        target_resp = torch.matmul(gate_c, target_contrib_t)
        side_db = 10.0 * torch.log10(torch.abs(af_grid).square().clamp_min(EPS))[side_mask_t]
        target_db = 10.0 * torch.log10(torch.abs(target_resp).square().clamp_min(EPS))
        smooth_side = temp * torch.logsumexp(side_db / temp, dim=0)
        weak = -temp * torch.logsumexp(-target_db / temp, dim=0)
        strong = temp * torch.logsumexp(target_db / temp, dim=0)
        spread = strong - weak
        loss = smooth_side - weak
        loss = loss + args.spread_weight * torch.relu(spread - args.spread_margin_db).square()
        loss = loss - args.weak_peak_weight * weak
        loss = loss + 0.01 * (gate * (1.0 - gate)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        soft = torch.sigmoid(logits)
        gate = soft * (float(num_active) / soft.sum().clamp_min(1.0e-6))
        idx = torch.topk(gate, k=num_active, largest=True).indices.cpu().numpy()
    mask = np.zeros(start_mask.shape[0], dtype=np.bool_)
    mask[idx] = True
    return mask


def run_greedy_from_mask(
    start_mask: np.ndarray,
    grid_contrib: np.ndarray,
    target_contrib: np.ndarray,
    side_mask: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
    max_swaps: int,
) -> dict[str, Any]:
    mask = start_mask.astype(bool).copy()
    af_grid = mask.astype(np.float32) @ grid_contrib
    target_resp = mask.astype(np.float32) @ target_contrib
    current_obj, initial_metrics = objective_from_fields(
        af_grid,
        target_resp,
        side_mask,
        spread_margin_db=args.spread_margin_db,
        spread_weight=args.spread_weight,
        weak_peak_weight=args.weak_peak_weight,
    )
    swaps = 0
    for _ in range(max_swaps):
        pairs = candidate_pairs(
            mask,
            grid_contrib,
            af_grid,
            side_mask,
            top_candidates=args.top_candidates,
            random_pairs=args.random_pairs,
            rng=rng,
        )
        best_obj, best_pair, _best_metrics = evaluate_pairs(
            pairs,
            af_grid,
            target_resp,
            grid_contrib,
            target_contrib,
            side_mask,
            args,
        )
        if best_pair is None or current_obj - best_obj < args.min_improve_db:
            break
        off, on = best_pair
        mask[off] = False
        mask[on] = True
        af_grid = af_grid - grid_contrib[off] + grid_contrib[on]
        target_resp = target_resp - target_contrib[off] + target_contrib[on]
        current_obj = best_obj
        swaps += 1
    final_obj, final_metrics = objective_from_fields(
        af_grid,
        target_resp,
        side_mask,
        spread_margin_db=args.spread_margin_db,
        spread_weight=args.spread_weight,
        weak_peak_weight=args.weak_peak_weight,
    )
    return {
        "mask": mask,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "swaps": swaps,
        "objective_improvement": float(initial_metrics["objective"]) - float(final_obj),
    }


def optimize_sample_v2(
    *,
    sample_index: int,
    arrays: dict[str, np.ndarray],
    initial_order: np.ndarray,
    grid_dirs: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    positions = arrays["positions_lambda"].astype(np.float32)
    num_active = int(arrays["num_active"][sample_index])
    num_elements = positions.shape[0]
    task_valid = arrays["task_valid"][sample_index].astype(bool)
    target_dirs = target_dirs_for_sample(arrays["targets_deg"][sample_index], task_valid)
    sample_rng = np.random.default_rng(
        stable_sample_seed(args, arrays["targets_deg"][sample_index], task_valid, num_active)
    )

    grid_contrib, target_contrib, target_tx = build_contributions(positions, target_dirs, grid_dirs, num_active)
    side_mask = side_mask_for_targets(grid_dirs, target_dirs)
    original_mask = arrays["masks"][sample_index].astype(bool).copy()
    original_obj, original_metrics = objective_from_fields(
        original_mask.astype(np.float32) @ grid_contrib,
        original_mask.astype(np.float32) @ target_contrib,
        side_mask,
        spread_margin_db=args.spread_margin_db,
        spread_weight=args.spread_weight,
        weak_peak_weight=args.weak_peak_weight,
    )

    seed_masks = canonical_seed_masks(
        element_ixiy=arrays["element_ixiy"],
        positions=positions,
        target_dirs=target_dirs,
        target_contrib=target_contrib,
        initial_order=initial_order,
        num_active=num_active,
        max_starts=args.v2_starts,
        rng=sample_rng,
    )
    candidate_masks = list(seed_masks)
    if num_active < num_elements:
        candidate_masks.extend(
            nsga_candidates(
                seed_masks,
                grid_contrib,
                target_contrib,
                side_mask,
                num_active,
                args,
                sample_rng,
            )
        )

    _, seed_scalar, _ = population_objectives(candidate_masks, grid_contrib, target_contrib, side_mask, args)
    best_seed = candidate_masks[int(np.argmin(seed_scalar))]
    if int(arrays["k_values"][sample_index]) >= args.softmask_min_k and num_active < num_elements:
        soft_candidate = softmask_gradient_candidate(
            best_seed,
            grid_contrib,
            target_contrib,
            side_mask,
            num_active,
            args,
        )
        if soft_candidate is not None:
            candidate_masks.append(soft_candidate)

    candidate_masks = unique_masks(candidate_masks)
    _, scalar, _ = population_objectives(candidate_masks, grid_contrib, target_contrib, side_mask, args)
    order = np.argsort(scalar, kind="stable")[: max(args.v2_final_candidates, 1)]
    final_runs = []
    for rank, candidate_index in enumerate(order):
        max_swaps = 0 if num_active >= num_elements else args.v2_greedy_swaps
        result = run_greedy_from_mask(
            candidate_masks[int(candidate_index)],
            grid_contrib,
            target_contrib,
            side_mask,
            args,
            sample_rng,
            max_swaps,
        )
        result["source_rank"] = int(rank)
        final_runs.append(result)
    best = min(final_runs, key=lambda item: float(item["final_metrics"]["objective"]))

    weights = np.zeros((num_elements, KMAX), dtype=np.complex64)
    valid_indices = np.flatnonzero(task_valid)
    weights[:, valid_indices] = target_tx * best["mask"][:, None].astype(np.complex64) / float(num_active)
    return {
        "sample_index": sample_index,
        "mask": best["mask"],
        "weights": weights,
        "initial_metrics": best["initial_metrics"],
        "final_metrics": best["final_metrics"],
        "original_metrics": original_metrics,
        "original_objective": float(original_obj),
        "swaps": int(best["swaps"]),
        "objective_improvement": float(best["objective_improvement"]),
        "v2_seed_count": len(seed_masks),
        "v2_candidate_count": len(candidate_masks),
        "v2_source_rank": int(best["source_rank"]),
        "v2_softmask_used": int(int(arrays["k_values"][sample_index]) >= args.softmask_min_k and num_active < num_elements),
    }


def optimize_sample(
    *,
    sample_index: int,
    arrays: dict[str, np.ndarray],
    initial_order: np.ndarray,
    grid_dirs: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.strategy == "v2":
        return optimize_sample_v2(
            sample_index=sample_index,
            arrays=arrays,
            initial_order=initial_order,
            grid_dirs=grid_dirs,
            rng=rng,
            args=args,
        )

    positions = arrays["positions_lambda"].astype(np.float32)
    num_active = int(arrays["num_active"][sample_index])
    num_elements = positions.shape[0]
    task_valid = arrays["task_valid"][sample_index].astype(bool)
    target_dirs = target_dirs_for_sample(arrays["targets_deg"][sample_index], task_valid)
    if args.init_mode == "original":
        mask = arrays["masks"][sample_index].astype(bool).copy()
    else:
        mask = initial_mask(initial_order, num_active, num_elements)
    if num_active >= num_elements:
        max_swaps = 0
    else:
        max_swaps = args.max_swaps

    grid_contrib, target_contrib, target_tx = build_contributions(positions, target_dirs, grid_dirs, num_active)
    side_mask = side_mask_for_targets(grid_dirs, target_dirs)
    af_grid = mask.astype(np.float32) @ grid_contrib
    target_resp = mask.astype(np.float32) @ target_contrib
    current_obj, initial_metrics = objective_from_fields(
        af_grid,
        target_resp,
        side_mask,
        spread_margin_db=args.spread_margin_db,
        spread_weight=args.spread_weight,
        weak_peak_weight=args.weak_peak_weight,
    )
    swaps = 0
    for _ in range(max_swaps):
        pairs = candidate_pairs(
            mask,
            grid_contrib,
            af_grid,
            side_mask,
            top_candidates=args.top_candidates,
            random_pairs=args.random_pairs,
            rng=rng,
        )
        best_obj, best_pair, _best_metrics = evaluate_pairs(
            pairs,
            af_grid,
            target_resp,
            grid_contrib,
            target_contrib,
            side_mask,
            args,
        )
        if best_pair is None or current_obj - best_obj < args.min_improve_db:
            break
        off, on = best_pair
        mask[off] = False
        mask[on] = True
        af_grid = af_grid - grid_contrib[off] + grid_contrib[on]
        target_resp = target_resp - target_contrib[off] + target_contrib[on]
        current_obj = best_obj
        swaps += 1

    final_obj, final_metrics = objective_from_fields(
        af_grid,
        target_resp,
        side_mask,
        spread_margin_db=args.spread_margin_db,
        spread_weight=args.spread_weight,
        weak_peak_weight=args.weak_peak_weight,
    )
    weights = np.zeros((num_elements, KMAX), dtype=np.complex64)
    valid_indices = np.flatnonzero(task_valid)
    weights[:, valid_indices] = target_tx * mask[:, None].astype(np.complex64) / float(num_active)
    return {
        "sample_index": sample_index,
        "mask": mask,
        "weights": weights,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "swaps": swaps,
        "objective_improvement": float(initial_obj := initial_metrics["objective"]) - float(final_obj),
    }


def compute_eval_metrics_for_masks(
    arrays: dict[str, np.ndarray],
    masks: np.ndarray,
    weights: np.ndarray,
    theta_step: float,
    phi_step: float,
) -> list[dict[str, float]]:
    positions = arrays["positions_lambda"].astype(np.float32)
    _, _, grid_dirs = make_grid(theta_step, phi_step)
    grid_phase = 2.0 * np.pi * (grid_dirs @ positions.T)
    grid_steer = np.exp(-1j * grid_phase).astype(np.complex64)
    rows: list[dict[str, float]] = []
    for i in range(masks.shape[0]):
        valid = arrays["task_valid"][i].astype(bool)
        target_dirs = target_dirs_for_sample(arrays["targets_deg"][i], valid)
        side_mask = side_mask_for_targets(grid_dirs, target_dirs)
        combined = weights[i].sum(axis=1)
        af_grid = grid_steer @ combined
        target_phase = 2.0 * np.pi * (target_dirs @ positions.T)
        target_steer = np.exp(-1j * target_phase).astype(np.complex64)
        target_resp = target_steer @ combined
        _obj, metrics = objective_from_fields(
            af_grid,
            target_resp,
            side_mask,
            spread_margin_db=5.0,
            spread_weight=0.0,
            weak_peak_weight=0.0,
        )
        rows.append(metrics)
    return rows


def write_hdf5_labels(
    dataset_path: Path,
    teacher_name: str,
    masks: np.ndarray,
    weights: np.ndarray,
    k_values: np.ndarray,
    metrics_rows: list[dict[str, Any]],
    *,
    overwrite: bool,
) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    num_samples, num_elements = masks.shape
    with h5py.File(dataset_path, "a") as handle:
        labels = handle.require_group("labels")
        if teacher_name in labels:
            if not overwrite:
                raise RuntimeError(f"labels/{teacher_name} exists; pass --overwrite to replace it.")
            del labels[teacher_name]
        group = labels.create_group(teacher_name)
        weights_ri = np.zeros((num_samples, num_elements, KMAX, 2), dtype=np.float64)
        weights_ri[..., 0] = weights.real
        weights_ri[..., 1] = weights.imag
        assignment = np.zeros((num_samples, num_elements, KMAX), dtype=np.float64)
        task_count = k_values.astype(np.int32)
        for i, k in enumerate(task_count):
            if k > 0:
                assignment[i, :, :k] = masks[i, :, None].astype(np.float64) / float(k)
        group.create_dataset("weights_real_imag", data=weights_ri)
        group.create_dataset("activation", data=masks.astype(np.float64))
        group.create_dataset("assignment", data=assignment)
        group.create_dataset("status", data=np.asarray(["ok"] * num_samples, dtype=object), dtype=string_dtype)
        group.create_dataset("solve_time_ms", data=np.zeros(num_samples, dtype=np.float64))
        group.create_dataset("e2e_time_ms", data=np.zeros(num_samples, dtype=np.float64))
        group.create_dataset("objective", data=np.asarray([r["objective"] for r in metrics_rows], dtype=np.float64))
        group.create_dataset("iterations", data=np.asarray([r["swaps"] for r in metrics_rows], dtype=np.int32))
        group.create_dataset("task_count", data=task_count)
        group.create_dataset("scenario_index", data=np.arange(num_samples, dtype=np.int32))
        diagnostics = [
            json.dumps(
                {
                    "teacher": teacher_name,
                    "strategy": r.get("strategy", "v1"),
                    "psll_to_weakest_peak_db": r["optimized_psll_to_weakest_peak_db"],
                    "weak_peak_db": r["optimized_weak_peak_db"],
                    "target_spread_db": r["optimized_spread_db"],
                    "swaps": r["swaps"],
                    "delta_psll_vs_original_db": r.get("delta_psll_vs_original_db"),
                },
                ensure_ascii=False,
            )
            for r in metrics_rows
        ]
        group.create_dataset("diagnostics_json", data=np.asarray(diagnostics, dtype=object), dtype=string_dtype)


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


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["k"], row["active_ratio"]), []).append(row)
    out: list[dict[str, Any]] = []
    for (k, ar), group in sorted(groups.items(), key=lambda x: (int(x[0][0]), float(x[0][1]))):
        def arr(key: str) -> np.ndarray:
            return np.asarray([float(r[key]) for r in group], dtype=np.float64)

        row = {
            "k": k,
            "active_ratio": ar,
            "n": len(group),
            "initial_psll_mean_db": float(arr("initial_psll_to_weakest_peak_db").mean()),
            "optimized_psll_mean_db": float(arr("optimized_psll_to_weakest_peak_db").mean()),
            "delta_psll_mean_db": float(arr("optimized_psll_to_weakest_peak_db").mean() - arr("initial_psll_to_weakest_peak_db").mean()),
            "initial_psll_p95_db": float(np.percentile(arr("initial_psll_to_weakest_peak_db"), 95)),
            "optimized_psll_p95_db": float(np.percentile(arr("optimized_psll_to_weakest_peak_db"), 95)),
            "initial_weak_peak_mean_db": float(arr("initial_weak_peak_db").mean()),
            "optimized_weak_peak_mean_db": float(arr("optimized_weak_peak_db").mean()),
            "delta_weak_peak_mean_db": float(arr("optimized_weak_peak_db").mean() - arr("initial_weak_peak_db").mean()),
            "mean_swaps": float(arr("swaps").mean()),
        }
        if "delta_psll_vs_original_db" in group[0]:
            row["original_psll_mean_db"] = float(arr("original_psll_to_weakest_peak_db").mean())
            row["delta_psll_vs_original_mean_db"] = float(arr("delta_psll_vs_original_db").mean())
            row["original_weak_peak_mean_db"] = float(arr("original_weak_peak_db").mean())
            row["delta_weak_peak_vs_original_mean_db"] = float(arr("delta_weak_peak_vs_original_db").mean())
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    dataset_dir: Path = args.dataset_dir
    out_dir = dataset_dir / "optimized_teachers" / args.teacher_name
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays_npz = np.load(dataset_dir / "dataset_arrays.npz", allow_pickle=False)
    arrays = {key: arrays_npz[key] for key in arrays_npz.files}
    sample_ids = arrays["sample_ids"]
    num_samples = int(arrays["k_values"].shape[0])
    if args.limit > 0:
        num_samples = min(num_samples, args.limit)

    theta_flat, phi_flat, grid_dirs = make_grid(args.theta_step, args.phi_step)
    order = deterministic_element_order(arrays["element_ixiy"])
    rng = np.random.default_rng(args.seed)
    masks = np.zeros((arrays["masks"].shape[0], arrays["masks"].shape[1]), dtype=np.int8)
    weights = np.zeros((arrays["masks"].shape[0], arrays["masks"].shape[1], KMAX), dtype=np.complex64)
    rows: list[dict[str, Any]] = []
    start = time.time()
    last = start
    for i in range(num_samples):
        result = optimize_sample(
            sample_index=i,
            arrays=arrays,
            initial_order=order,
            grid_dirs=grid_dirs,
            rng=rng,
            args=args,
        )
        masks[i] = result["mask"].astype(np.int8)
        weights[i] = result["weights"]
        initial = result["initial_metrics"]
        final = result["final_metrics"]
        original = result.get("original_metrics", initial)
        row = {
            "sample_index": i,
            "sample_id": str(sample_ids[i]),
            "k": int(arrays["k_values"][i]),
            "active_ratio": f"{float(arrays['active_ratios_requested'][i]):.1f}",
            "num_active": int(arrays["num_active"][i]),
            "swaps": int(result["swaps"]),
            "strategy": args.strategy,
            "v2_seed_count": int(result.get("v2_seed_count", 0)),
            "v2_candidate_count": int(result.get("v2_candidate_count", 0)),
            "v2_source_rank": int(result.get("v2_source_rank", -1)),
            "v2_softmask_used": int(result.get("v2_softmask_used", 0)),
            "original_psll_to_weakest_peak_db": original["psll_to_weakest_peak_db"],
            "original_weak_peak_db": original["weak_peak_db"],
            "initial_psll_to_weakest_peak_db": initial["psll_to_weakest_peak_db"],
            "optimized_psll_to_weakest_peak_db": final["psll_to_weakest_peak_db"],
            "delta_psll_to_weakest_peak_db": final["psll_to_weakest_peak_db"] - initial["psll_to_weakest_peak_db"],
            "delta_psll_vs_original_db": final["psll_to_weakest_peak_db"] - original["psll_to_weakest_peak_db"],
            "initial_weak_peak_db": initial["weak_peak_db"],
            "optimized_weak_peak_db": final["weak_peak_db"],
            "delta_weak_peak_db": final["weak_peak_db"] - initial["weak_peak_db"],
            "delta_weak_peak_vs_original_db": final["weak_peak_db"] - original["weak_peak_db"],
            "initial_spread_db": initial["target_spread_db"],
            "optimized_spread_db": final["target_spread_db"],
            "objective": final["objective"],
        }
        rows.append(row)
        now = time.time()
        if now - last > 20.0 or i + 1 == num_samples:
            print(f"optimized {i + 1}/{num_samples} samples, elapsed {now - start:.1f}s")
            last = now

    # For untouched samples in --limit smoke tests, keep original labels outside the limit.
    if num_samples < arrays["masks"].shape[0]:
        masks[num_samples:] = arrays["masks"][num_samples:].astype(np.int8)
        original_weights = arrays["task_weights_real_imag"][num_samples:, :, :, 0] + 1j * arrays["task_weights_real_imag"][num_samples:, :, :, 1]
        weights[num_samples:] = original_weights.astype(np.complex64)

    np.savez_compressed(
        out_dir / "optimized_teacher_arrays.npz",
        masks=masks,
        task_weights_real_imag=np.stack([weights.real, weights.imag], axis=-1),
        sample_ids=arrays["sample_ids"],
        k_values=arrays["k_values"],
        active_ratios_requested=arrays["active_ratios_requested"],
        active_ratios_actual=arrays["active_ratios_actual"],
        num_active=arrays["num_active"],
        targets_deg=arrays["targets_deg"],
        task_valid=arrays["task_valid"],
        positions_lambda=arrays["positions_lambda"],
        element_ixiy=arrays["element_ixiy"],
    )
    # Also create a dataset_arrays.npz-compatible copy for direct training.
    compatible = {key: arrays[key] for key in arrays.files} if hasattr(arrays, "files") else dict(arrays)
    compatible["masks"] = masks
    compatible["task_weights_real_imag"] = np.stack([weights.real, weights.imag], axis=-1)
    np.savez_compressed(out_dir / "dataset_arrays.npz", **compatible)
    shutil.copy2(dataset_dir / "training_split_manifest.json", out_dir / "training_split_manifest.json")

    write_csv(out_dir / "optimized_teacher_metrics.csv", rows)
    summary_rows = aggregate(rows)
    write_csv(out_dir / "optimized_teacher_summary_by_k_active.csv", summary_rows)
    hdf5_label = None
    if num_samples == arrays["masks"].shape[0]:
        write_hdf5_labels(
            dataset_dir / "training_dataset.h5",
            args.teacher_name,
            masks.astype(np.float64),
            weights,
            arrays["k_values"],
            rows,
            overwrite=args.overwrite,
        )
        hdf5_label = f"{dataset_dir / 'training_dataset.h5'}:/labels/{args.teacher_name}"
    run_summary = {
        "teacher_name": args.teacher_name,
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "num_samples": num_samples,
        "elapsed_s": time.time() - start,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "outputs": {
            "optimized_teacher_arrays": str(out_dir / "optimized_teacher_arrays.npz"),
            "dataset_arrays": str(out_dir / "dataset_arrays.npz"),
            "metrics": str(out_dir / "optimized_teacher_metrics.csv"),
            "summary": str(out_dir / "optimized_teacher_summary_by_k_active.csv"),
            "hdf5_label": hdf5_label,
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
