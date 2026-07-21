"""Generate URA16 multi-task activation/weight samples for HFSS post-processing.

The generated data follows the current quick HFSS model:
- 16 x 16 centered URA, element index = ix * ny + iy.
- Ports are named P000 ... P255.
- Element spacing is lambda / 2 in both x and y.

Each sample contains:
- random target directions for K in {1, 2, 4, 6};
- an activation mask matching one requested active ratio;
- per-task steering weights;
- one combined HFSS port excitation vector for exporting a multi-beam pattern.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

LOCAL_DEPS = Path(__file__).resolve().parents[1] / ".python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "hfss_outputs" / "multitask_dataset"

K_CHOICES = (1, 2, 4, 6)
ACTIVE_RATIOS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
KMAX = max(K_CHOICES)


@dataclass(frozen=True)
class ArrayConfig:
    nx: int = 16
    ny: int = 16
    dx_lambda: float = 0.5
    dy_lambda: float = 0.5
    wavelength: float = 1.0

    @property
    def num_elements(self) -> int:
        return self.nx * self.ny


@dataclass(frozen=True)
class GeneratorConfig:
    samples_per_combo: int
    seed: int
    theta_min_deg: float
    theta_max_deg: float
    phi_min_deg: float
    phi_max_deg: float
    min_separation_deg: float
    array: ArrayConfig


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_clean:
        clean_output_dir(out_dir)

    array_cfg = ArrayConfig()
    config = GeneratorConfig(
        samples_per_combo=int(args.samples_per_combo),
        seed=int(args.seed),
        theta_min_deg=float(args.theta_min_deg),
        theta_max_deg=float(args.theta_max_deg),
        phi_min_deg=float(args.phi_min_deg),
        phi_max_deg=float(args.phi_max_deg),
        min_separation_deg=float(args.min_separation_deg),
        array=array_cfg,
    )

    rng = np.random.default_rng(config.seed)
    positions = build_ura_positions(array_cfg)
    element_rows = build_element_rows(array_cfg, positions)
    samples = generate_samples(config=config, rng=rng, positions=positions, out_dir=out_dir, element_rows=element_rows)

    write_dataset_npz(out_dir / "dataset_arrays.npz", samples=samples, element_rows=element_rows, positions=positions)
    write_training_hdf5(out_dir / "training_dataset.h5", samples=samples, config=config, positions=positions)
    write_split_manifest(out_dir / "training_split_manifest.json", samples=samples, seed=config.seed)
    write_manifest(out_dir / "manifest.csv", samples)
    write_export_queue(out_dir / "hfss_export_queue.csv", samples)
    write_config(out_dir / "dataset_config.json", config=config)
    write_hfss_export_vbs(out_dir / "export_hfss_patterns.vbs", out_dir=out_dir)
    write_hfss_export_ps1(out_dir / "run_hfss_export_patterns.ps1", out_dir=out_dir)
    write_readme(out_dir / "README_multitask_dataset.md", out_dir=out_dir, samples=samples, config=config)

    print(f"Generated {len(samples)} samples")
    print(f"Output: {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.csv'}")
    print(f"NPZ: {out_dir / 'dataset_arrays.npz'}")
    print(f"Training HDF5: {out_dir / 'training_dataset.h5'}")
    print(f"Split manifest: {out_dir / 'training_split_manifest.json'}")
    print(f"HFSS export queue: {out_dir / 'hfss_export_queue.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-task URA16 mask/weight samples for HFSS.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output dataset directory.")
    parser.add_argument("--samples-per-combo", type=int, default=1, help="Samples for each K x active_ratio pair.")
    parser.add_argument("--seed", type=int, default=20260625, help="Random seed.")
    parser.add_argument("--theta-min-deg", type=float, default=0.0)
    parser.add_argument("--theta-max-deg", type=float, default=60.0)
    parser.add_argument("--phi-min-deg", type=float, default=0.0)
    parser.add_argument("--phi-max-deg", type=float, default=360.0)
    parser.add_argument("--min-separation-deg", type=float, default=10.0)
    parser.add_argument("--no-clean", action="store_true", help="Do not clean old generated files in the output directory.")
    return parser.parse_args()


def clean_output_dir(out_dir: Path) -> None:
    samples_dir = out_dir / "samples"
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    for name in (
        "dataset_arrays.npz",
        "dataset_config.json",
        "export_hfss_patterns.log",
        "export_hfss_patterns.vbs",
        "hfss_export_queue.csv",
        "manifest.csv",
        "README_multitask_dataset.md",
        "run_hfss_export_patterns.ps1",
        "training_dataset.h5",
        "training_split_manifest.json",
    ):
        path = out_dir / name
        if path.exists():
            path.unlink()


def generate_samples(
    *,
    config: GeneratorConfig,
    rng: np.random.Generator,
    positions: np.ndarray,
    out_dir: Path,
    element_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    sample_index = 0

    for k in K_CHOICES:
        for active_ratio in ACTIVE_RATIOS:
            for combo_index in range(config.samples_per_combo):
                sample_id = f"sample_{sample_index:06d}"
                sample_dir = out_dir / "samples" / sample_id
                sample_dir.mkdir(parents=True, exist_ok=True)

                targets = sample_targets(
                    rng=rng,
                    k=k,
                    theta_min=config.theta_min_deg,
                    theta_max=config.theta_max_deg,
                    phi_min=config.phi_min_deg,
                    phi_max=config.phi_max_deg,
                    min_sep=config.min_separation_deg,
                )
                mask = sample_active_mask(rng=rng, num_elements=config.array.num_elements, active_ratio=active_ratio)
                task_weights = compute_task_weights(
                    positions=positions,
                    targets=targets,
                    mask=mask,
                    wavelength=config.array.wavelength,
                )
                hfss_weights = combine_task_weights_for_hfss(task_weights)

                paths = write_sample_files(
                    sample_dir=sample_dir,
                    sample_id=sample_id,
                    k=k,
                    active_ratio=active_ratio,
                    combo_index=combo_index,
                    targets=targets,
                    mask=mask,
                    task_weights=task_weights,
                    hfss_weights=hfss_weights,
                    element_rows=element_rows,
                )

                sample = {
                    "sample_id": sample_id,
                    "sample_index": sample_index,
                    "combo_index": combo_index,
                    "k": k,
                    "active_ratio_requested": float(active_ratio),
                    "active_ratio_actual": float(mask.mean()),
                    "num_active": int(mask.sum()),
                    "targets": targets.tolist(),
                    "mask": mask.astype(np.int8),
                    "task_weights": task_weights,
                    "hfss_weights": hfss_weights,
                    "sample_dir": str(sample_dir),
                    **paths,
                }
                samples.append(sample)
                sample_index += 1

    return samples


def build_ura_positions(array_cfg: ArrayConfig) -> np.ndarray:
    x_coords = (np.arange(array_cfg.nx, dtype=np.float64) - 0.5 * (array_cfg.nx - 1))
    y_coords = (np.arange(array_cfg.ny, dtype=np.float64) - 0.5 * (array_cfg.ny - 1))
    x_coords *= array_cfg.dx_lambda * array_cfg.wavelength
    y_coords *= array_cfg.dy_lambda * array_cfg.wavelength
    xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")
    zz = np.zeros_like(xx)
    return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))


def build_element_rows(array_cfg: ArrayConfig, positions: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ix in range(array_cfg.nx):
        for iy in range(array_cfg.ny):
            element_index = ix * array_cfg.ny + iy
            x_lambda, y_lambda, z_lambda = positions[element_index]
            rows.append(
                {
                    "element_index": element_index,
                    "port_name": f"P{element_index:03d}",
                    "ix": ix,
                    "iy": iy,
                    "x_lambda": float(x_lambda),
                    "y_lambda": float(y_lambda),
                    "z_lambda": float(z_lambda),
                }
            )
    return rows


def sample_targets(
    *,
    rng: np.random.Generator,
    k: int,
    theta_min: float,
    theta_max: float,
    phi_min: float,
    phi_max: float,
    min_sep: float,
) -> np.ndarray:
    targets: list[tuple[float, float]] = []
    max_attempts = 10000

    for _ in range(max_attempts):
        theta = float(rng.uniform(theta_min, theta_max))
        phi = wrap_phi(float(rng.uniform(phi_min, phi_max)))
        if all(great_circle_distance_deg(theta, phi, old_theta, old_phi) >= min_sep for old_theta, old_phi in targets):
            targets.append((theta, phi))
            if len(targets) == k:
                return np.asarray(targets, dtype=np.float64)

    raise RuntimeError(f"Could not sample {k} targets with {min_sep} deg minimum separation.")


def sample_active_mask(*, rng: np.random.Generator, num_elements: int, active_ratio: float) -> np.ndarray:
    requested = int(round(float(active_ratio) * num_elements))
    num_active = min(num_elements, max(1, requested))
    mask = np.zeros(num_elements, dtype=bool)
    if num_active == num_elements:
        mask[:] = True
        return mask
    active_indices = rng.choice(num_elements, size=num_active, replace=False)
    mask[active_indices] = True
    return mask


def compute_task_weights(
    *,
    positions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    wavelength: float,
) -> np.ndarray:
    num_elements = positions.shape[0]
    num_tasks = targets.shape[0]
    weights = np.zeros((num_elements, num_tasks), dtype=np.complex128)
    num_active = max(int(mask.sum()), 1)

    for task_index, (theta_deg, phi_deg) in enumerate(targets):
        steering = steering_vector(
            positions=positions,
            theta_deg=float(theta_deg),
            phi_deg=float(phi_deg),
            wavelength=wavelength,
        )
        weights[:, task_index] = mask.astype(np.float64) * steering / float(num_active)

    return weights


def combine_task_weights_for_hfss(task_weights: np.ndarray) -> np.ndarray:
    # The beam-multitask code uses weights with the response convention
    # A(theta, phi) @ w, where A contains exp(-j k r.u). HFSS source phases
    # follow the opposite propagation sign, so the port excitation vector must
    # be conjugated to point at the same target direction.
    combined = np.conjugate(np.sum(task_weights, axis=1))
    l2_norm = float(np.linalg.norm(combined))
    if l2_norm <= 0.0:
        return combined
    # Unit L2 norm means one watt total incident power in Driven Modal.
    return combined / l2_norm


def steering_vector(*, positions: np.ndarray, theta_deg: float, phi_deg: float, wavelength: float) -> np.ndarray:
    direction = angles_to_unit_vector(theta_deg=theta_deg, phi_deg=phi_deg)
    phase = (2.0 * np.pi / wavelength) * (positions @ direction)
    return np.exp(1j * phase).astype(np.complex128, copy=False)


def angles_to_unit_vector(*, theta_deg: float, phi_deg: float) -> np.ndarray:
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    sin_theta = math.sin(theta)
    return np.asarray(
        [sin_theta * math.cos(phi), sin_theta * math.sin(phi), math.cos(theta)],
        dtype=np.float64,
    )


def great_circle_distance_deg(theta_a: float, phi_a: float, theta_b: float, phi_b: float) -> float:
    ua = angles_to_unit_vector(theta_deg=theta_a, phi_deg=phi_a)
    ub = angles_to_unit_vector(theta_deg=theta_b, phi_deg=phi_b)
    cosine = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def wrap_phi(phi_deg: float) -> float:
    wrapped = phi_deg % 360.0
    if wrapped < 0.0:
        wrapped += 360.0
    return wrapped


def write_sample_files(
    *,
    sample_dir: Path,
    sample_id: str,
    k: int,
    active_ratio: float,
    combo_index: int,
    targets: np.ndarray,
    mask: np.ndarray,
    task_weights: np.ndarray,
    hfss_weights: np.ndarray,
    element_rows: list[dict[str, object]],
) -> dict[str, str]:
    targets_csv = sample_dir / "targets.csv"
    mask_csv = sample_dir / "mask.csv"
    weights_csv = sample_dir / "weights.csv"
    sources_csv = sample_dir / "hfss_sources.csv"
    sample_json = sample_dir / "sample.json"

    with targets_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_index", "theta_deg", "phi_deg", "gain_target_db", "priority"])
        for task_index, (theta, phi) in enumerate(targets):
            writer.writerow([task_index, f"{theta:.6f}", f"{phi:.6f}", "0.000000", "1.000000"])

    with mask_csv.open("w", newline="") as f:
        fieldnames = ["element_index", "port_name", "ix", "iy", "x_lambda", "y_lambda", "z_lambda", "active"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in element_rows:
            idx = int(row["element_index"])
            out = dict(row)
            out["active"] = int(mask[idx])
            writer.writerow(out)

    weight_fieldnames = [
        "element_index",
        "port_name",
        "ix",
        "iy",
        "active",
        "source_convention",
        "hfss_field_coefficient_magnitude",
        "hfss_incident_power_w",
        "hfss_magnitude_v",
        "hfss_phase_deg",
        "hfss_real",
        "hfss_imag",
    ]
    for task_index in range(KMAX):
        weight_fieldnames.extend(
            [
                f"task{task_index}_real",
                f"task{task_index}_imag",
                f"task{task_index}_magnitude",
                f"task{task_index}_phase_deg",
            ]
        )

    with weights_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=weight_fieldnames)
        writer.writeheader()
        for row in element_rows:
            idx = int(row["element_index"])
            hfss = hfss_weights[idx]
            out = {
                "element_index": idx,
                "port_name": row["port_name"],
                "ix": row["ix"],
                "iy": row["iy"],
                "active": int(mask[idx]),
                "source_convention": "driven_modal_incident_power_watt",
                "hfss_field_coefficient_magnitude": f"{abs(hfss):.9f}",
                "hfss_incident_power_w": f"{abs(hfss) ** 2:.12e}",
                # Retained for readers of legacy files; this is a field coefficient, not volts.
                "hfss_magnitude_v": f"{abs(hfss):.9f}",
                "hfss_phase_deg": f"{phase_deg(hfss):.6f}",
                "hfss_real": f"{hfss.real:.12e}",
                "hfss_imag": f"{hfss.imag:.12e}",
            }
            for task_index in range(KMAX):
                value = task_weights[idx, task_index] if task_index < k else 0.0 + 0.0j
                out[f"task{task_index}_real"] = f"{value.real:.12e}"
                out[f"task{task_index}_imag"] = f"{value.imag:.12e}"
                out[f"task{task_index}_magnitude"] = f"{abs(value):.12e}"
                out[f"task{task_index}_phase_deg"] = f"{phase_deg(value):.6f}"
            writer.writerow(out)

    with sources_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PortName", "IncidentPowerW", "PhaseDeg", "Active", "ElementIndex"])
        for row in element_rows:
            idx = int(row["element_index"])
            value = hfss_weights[idx]
            writer.writerow(
                [
                    row["port_name"],
                    f"{abs(value) ** 2:.12e}",
                    f"{phase_deg(value):.6f}",
                    int(mask[idx]),
                    idx,
                ]
            )

    active_indices = np.flatnonzero(mask).astype(int).tolist()
    payload = {
        "sample_id": sample_id,
        "k": k,
        "combo_index": combo_index,
        "active_ratio_requested": float(active_ratio),
        "active_ratio_actual": float(mask.mean()),
        "num_active": int(mask.sum()),
        "targets": [
            {"task_index": int(i), "theta_deg": float(theta), "phi_deg": float(phi)}
            for i, (theta, phi) in enumerate(targets)
        ],
        "active_indices": active_indices,
        "source_convention": "driven_modal_incident_power_watt",
        "source_mapping": "incident_power_w=abs(complex_field_coefficient)**2",
        "total_incident_power_w": float(np.sum(np.abs(hfss_weights) ** 2)),
        "files": {
            "targets_csv": str(targets_csv),
            "mask_csv": str(mask_csv),
            "weights_csv": str(weights_csv),
            "hfss_sources_csv": str(sources_csv),
        },
    }
    sample_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "targets_csv": str(targets_csv),
        "mask_csv": str(mask_csv),
        "weights_csv": str(weights_csv),
        "hfss_sources_csv": str(sources_csv),
        "sample_json": str(sample_json),
    }


def write_dataset_npz(
    path: Path,
    *,
    samples: list[dict[str, object]],
    element_rows: list[dict[str, object]],
    positions: np.ndarray,
) -> None:
    num_samples = len(samples)
    num_elements = len(element_rows)

    masks = np.zeros((num_samples, num_elements), dtype=np.int8)
    targets = np.full((num_samples, KMAX, 2), np.nan, dtype=np.float64)
    task_valid = np.zeros((num_samples, KMAX), dtype=np.int8)
    task_weights = np.zeros((num_samples, num_elements, KMAX, 2), dtype=np.float64)
    hfss_weights = np.zeros((num_samples, num_elements, 2), dtype=np.float64)
    hfss_magnitude_v = np.zeros((num_samples, num_elements), dtype=np.float64)
    hfss_incident_power_w = np.zeros((num_samples, num_elements), dtype=np.float64)
    hfss_phase_deg = np.zeros((num_samples, num_elements), dtype=np.float64)
    k_values = np.zeros((num_samples,), dtype=np.int64)
    active_ratios_requested = np.zeros((num_samples,), dtype=np.float64)
    active_ratios_actual = np.zeros((num_samples,), dtype=np.float64)
    num_active = np.zeros((num_samples,), dtype=np.int64)
    sample_ids = np.empty((num_samples,), dtype=f"<U{len('sample_000000')}")

    for row_index, sample in enumerate(samples):
        k = int(sample["k"])
        mask = np.asarray(sample["mask"], dtype=np.int8)
        sample_task_weights = np.asarray(sample["task_weights"], dtype=np.complex128)
        sample_hfss_weights = np.asarray(sample["hfss_weights"], dtype=np.complex128)
        sample_targets = np.asarray(sample["targets"], dtype=np.float64)

        masks[row_index] = mask
        targets[row_index, :k, :] = sample_targets
        task_valid[row_index, :k] = 1
        task_weights[row_index, :, :k, 0] = sample_task_weights.real
        task_weights[row_index, :, :k, 1] = sample_task_weights.imag
        hfss_weights[row_index, :, 0] = sample_hfss_weights.real
        hfss_weights[row_index, :, 1] = sample_hfss_weights.imag
        hfss_magnitude_v[row_index] = np.abs(sample_hfss_weights)
        hfss_incident_power_w[row_index] = np.abs(sample_hfss_weights) ** 2
        hfss_phase_deg[row_index] = np.asarray([phase_deg(v) for v in sample_hfss_weights], dtype=np.float64)
        k_values[row_index] = k
        active_ratios_requested[row_index] = float(sample["active_ratio_requested"])
        active_ratios_actual[row_index] = float(sample["active_ratio_actual"])
        num_active[row_index] = int(sample["num_active"])
        sample_ids[row_index] = str(sample["sample_id"])

    np.savez_compressed(
        path,
        sample_ids=sample_ids,
        k_values=k_values,
        active_ratios_requested=active_ratios_requested,
        active_ratios_actual=active_ratios_actual,
        num_active=num_active,
        targets_deg=targets,
        task_valid=task_valid,
        masks=masks,
        task_weights_real_imag=task_weights,
        hfss_weights_real_imag=hfss_weights,
        hfss_magnitude_v=hfss_magnitude_v,
        hfss_incident_power_w=hfss_incident_power_w,
        hfss_phase_deg=hfss_phase_deg,
        positions_lambda=positions,
        port_names=np.asarray([str(row["port_name"]) for row in element_rows]),
        element_ixiy=np.asarray([[int(row["ix"]), int(row["iy"])] for row in element_rows], dtype=np.int64),
    )


def write_training_hdf5(
    path: Path,
    *,
    samples: list[dict[str, object]],
    config: GeneratorConfig,
    positions: np.ndarray,
) -> None:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required to write training_dataset.h5. "
            "Install it locally with: python -m pip install --target .python_deps h5py"
        ) from exc

    num_samples = len(samples)
    num_elements = config.array.num_elements
    string_dtype = h5py.string_dtype(encoding="utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        meta = handle.create_group("meta")
        meta.attrs["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        meta.attrs["package_version"] = "hfss_ura16_quick_model"
        meta.attrs["git_hash"] = "unknown"
        meta.attrs["git_dirty"] = False
        meta.attrs["runtime_json"] = json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "generator": str(Path(__file__).resolve()),
            },
            ensure_ascii=False,
        )
        meta.attrs["config_snapshot_json"] = json.dumps(asdict(config), ensure_ascii=False)
        meta.attrs["extra_meta_json"] = json.dumps(
            {
                "source": "hfss_ura16_quick_model",
                "label_solver": "hfss_sparse_steering",
                "hfss_source_convention": "conjugate(sum(task_weights)) for AEDT port excitation",
            },
            ensure_ascii=False,
        )
        meta.attrs["array_spec_json"] = json.dumps(
            {
                "array_id": "ura16",
                "nx": config.array.nx,
                "ny": config.array.ny,
                "dx": config.array.dx_lambda,
                "dy": config.array.dy_lambda,
                "wavelength": config.array.wavelength,
                "tile_shape": [4, 4],
            },
            ensure_ascii=False,
        )
        meta.attrs["grid_spec_json"] = json.dumps(
            {
                "theta_min_deg": 0.0,
                "theta_max_deg": 90.0,
                "theta_step_deg": 1.0,
                "phi_min_deg": 0.0,
                "phi_max_deg": 360.0,
                "phi_step_deg": 2.0,
                "mainlobe_radius_deg": 8.0,
                "minimum_task_separation_deg": config.min_separation_deg,
                "hpbw_cut_mode": "phi_fixed",
            },
            ensure_ascii=False,
        )

        array_group = handle.create_group("array")
        array_group.create_dataset("positions_xyz", data=positions)
        array_group.create_dataset("x", data=positions[:, 0])
        array_group.create_dataset("y", data=positions[:, 1])
        array_group.create_dataset("z", data=positions[:, 2])

        grid_group = handle.create_group("grid")
        grid_group.create_dataset("theta_values_deg", data=np.arange(0.0, 90.0 + 0.5, 1.0, dtype=np.float64))
        grid_group.create_dataset("phi_values_deg", data=np.arange(0.0, 360.0, 2.0, dtype=np.float64))

        scenarios_group = handle.create_group("scenarios")
        task_theta = np.full((num_samples, KMAX), np.nan, dtype=np.float64)
        task_phi = np.full((num_samples, KMAX), np.nan, dtype=np.float64)
        task_priority = np.full((num_samples, KMAX), np.nan, dtype=np.float64)
        task_gain_target = np.full((num_samples, KMAX), np.nan, dtype=np.float64)
        task_sll_mask = np.full((num_samples, KMAX), np.nan, dtype=np.float64)
        task_mask = np.zeros((num_samples, KMAX), dtype=np.bool_)

        share_mode = np.empty((num_samples,), dtype=object)
        environment_mode = np.empty((num_samples,), dtype=object)
        total_power_max = np.ones((num_samples,), dtype=np.float64)
        max_active_ratio = np.zeros((num_samples,), dtype=np.float64)
        noise_var = np.full((num_samples,), 1.0e-3, dtype=np.float64)
        num_snapshots = np.full((num_samples,), 256, dtype=np.int32)
        sigma_phase_deg = np.zeros((num_samples,), dtype=np.float64)
        sigma_amp_db = np.zeros((num_samples,), dtype=np.float64)
        element_failure_rate = np.zeros((num_samples,), dtype=np.float64)
        budget_max_solve_time_ms = np.full((num_samples,), np.nan, dtype=np.float64)
        budget_max_e2e_time_ms = np.full((num_samples,), np.nan, dtype=np.float64)
        metadata_json = np.empty((num_samples,), dtype=object)

        weights_real_imag = np.zeros((num_samples, num_elements, KMAX, 2), dtype=np.float64)
        activation = np.zeros((num_samples, num_elements), dtype=np.float64)
        assignment = np.zeros((num_samples, num_elements, KMAX), dtype=np.float64)
        status = np.empty((num_samples,), dtype=object)
        solve_time_ms = np.zeros((num_samples,), dtype=np.float64)
        e2e_time_ms = np.zeros((num_samples,), dtype=np.float64)
        objective = np.full((num_samples,), np.nan, dtype=np.float64)
        iterations = np.full((num_samples,), -1, dtype=np.int32)
        task_count = np.zeros((num_samples,), dtype=np.int32)
        scenario_index = np.arange(num_samples, dtype=np.int32)
        diagnostics_json = np.empty((num_samples,), dtype=object)

        hfss_group = handle.create_group("hfss")
        hfss_weights_real_imag = np.zeros((num_samples, num_elements, 2), dtype=np.float64)
        hfss_magnitude_v = np.zeros((num_samples, num_elements), dtype=np.float64)
        hfss_incident_power_w = np.zeros((num_samples, num_elements), dtype=np.float64)
        hfss_phase_deg = np.zeros((num_samples, num_elements), dtype=np.float64)

        for i, sample in enumerate(samples):
            k = int(sample["k"])
            targets = np.asarray(sample["targets"], dtype=np.float64)
            mask = np.asarray(sample["mask"], dtype=np.float64)
            task_w = np.asarray(sample["task_weights"], dtype=np.complex128)
            hfss_w = np.asarray(sample["hfss_weights"], dtype=np.complex128)
            ratio = float(sample["active_ratio_requested"])

            task_theta[i, :k] = targets[:, 0]
            task_phi[i, :k] = targets[:, 1]
            task_priority[i, :k] = 1.0
            task_gain_target[i, :k] = 0.0
            task_sll_mask[i, :k] = -20.0
            task_mask[i, :k] = True

            share_mode[i] = "shared"
            environment_mode[i] = "pattern_only"
            max_active_ratio[i] = ratio
            metadata_json[i] = json.dumps(
                {
                    "sample_id": sample["sample_id"],
                    "sample_index": int(sample["sample_index"]),
                    "active_ratio_requested": ratio,
                    "active_ratio_actual": float(sample["active_ratio_actual"]),
                    "num_active": int(sample["num_active"]),
                    "targets": [
                        {"task_index": int(j), "theta_deg": float(theta), "phi_deg": float(phi)}
                        for j, (theta, phi) in enumerate(targets)
                    ],
                    "hfss_sources_csv": sample["hfss_sources_csv"],
                },
                ensure_ascii=False,
            )

            weights_real_imag[i, :, :k, 0] = task_w.real
            weights_real_imag[i, :, :k, 1] = task_w.imag
            activation[i] = mask
            assignment[i, :, :k] = mask[:, None] / float(max(k, 1))
            status[i] = "ok"
            task_count[i] = k
            diagnostics_json[i] = json.dumps(
                {
                    "solver": "hfss_sparse_steering",
                    "active_ratio_requested": ratio,
                    "active_ratio_actual": float(sample["active_ratio_actual"]),
                    "num_active": int(sample["num_active"]),
                    "weight_convention": "beamlearn steering weights; HFSS sources are conjugated separately",
                },
                ensure_ascii=False,
            )

            hfss_weights_real_imag[i, :, 0] = hfss_w.real
            hfss_weights_real_imag[i, :, 1] = hfss_w.imag
            hfss_magnitude_v[i] = np.abs(hfss_w)
            hfss_incident_power_w[i] = np.abs(hfss_w) ** 2
            hfss_phase_deg[i] = np.asarray([phase_deg(v) for v in hfss_w], dtype=np.float64)

        scenarios_group.create_dataset("task_theta_deg", data=task_theta)
        scenarios_group.create_dataset("task_phi_deg", data=task_phi)
        scenarios_group.create_dataset("task_priority", data=task_priority)
        scenarios_group.create_dataset("task_gain_target_db", data=task_gain_target)
        scenarios_group.create_dataset("task_sll_mask_db", data=task_sll_mask)
        scenarios_group.create_dataset("task_mask", data=task_mask)
        scenarios_group.create_dataset("share_mode", data=share_mode, dtype=string_dtype)
        scenarios_group.create_dataset("environment_mode", data=environment_mode, dtype=string_dtype)
        scenarios_group.create_dataset("total_power_max", data=total_power_max)
        scenarios_group.create_dataset("max_active_ratio", data=max_active_ratio)
        scenarios_group.create_dataset("noise_var", data=noise_var)
        scenarios_group.create_dataset("num_snapshots", data=num_snapshots)
        scenarios_group.create_dataset("sigma_phase_deg", data=sigma_phase_deg)
        scenarios_group.create_dataset("sigma_amp_db", data=sigma_amp_db)
        scenarios_group.create_dataset("element_failure_rate", data=element_failure_rate)
        scenarios_group.create_dataset("budget_max_solve_time_ms", data=budget_max_solve_time_ms)
        scenarios_group.create_dataset("budget_max_e2e_time_ms", data=budget_max_e2e_time_ms)
        scenarios_group.create_dataset("metadata_json", data=metadata_json, dtype=string_dtype)

        labels_group = handle.create_group("labels")
        solver_group = labels_group.create_group("hfss_sparse_steering")
        solver_group.create_dataset("weights_real_imag", data=weights_real_imag)
        solver_group.create_dataset("activation", data=activation)
        solver_group.create_dataset("assignment", data=assignment)
        solver_group.create_dataset("status", data=status, dtype=string_dtype)
        solver_group.create_dataset("solve_time_ms", data=solve_time_ms)
        solver_group.create_dataset("e2e_time_ms", data=e2e_time_ms)
        solver_group.create_dataset("objective", data=objective)
        solver_group.create_dataset("iterations", data=iterations)
        solver_group.create_dataset("task_count", data=task_count)
        solver_group.create_dataset("scenario_index", data=scenario_index)
        solver_group.create_dataset("diagnostics_json", data=diagnostics_json, dtype=string_dtype)

        hfss_group.create_dataset("weights_real_imag", data=hfss_weights_real_imag)
        hfss_group.create_dataset("magnitude_v", data=hfss_magnitude_v)
        hfss_group.create_dataset("incident_power_w", data=hfss_incident_power_w)
        hfss_group.create_dataset("phase_deg", data=hfss_phase_deg)


def write_split_manifest(path: Path, *, samples: list[dict[str, object]], seed: int) -> None:
    rng = np.random.default_rng(seed + 17)
    groups: dict[tuple[int, str], list[int]] = {}
    for index, sample in enumerate(samples):
        key = (int(sample["k"]), f"{float(sample['active_ratio_requested']):.3f}")
        groups.setdefault(key, []).append(index)

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []

    for indices in groups.values():
        shuffled = np.asarray(indices, dtype=np.int64)
        rng.shuffle(shuffled)
        n = shuffled.size
        n_train = int(np.floor(0.70 * n))
        n_val = int(np.floor(0.15 * n))
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            if n_train + n_val >= n:
                n_val = max(0, n - n_train - 1)
        elif n == 2:
            n_train, n_val = 1, 0
        elif n == 1:
            n_train, n_val = 1, 0
        train.extend(int(x) for x in shuffled[:n_train])
        val.extend(int(x) for x in shuffled[n_train : n_train + n_val])
        test.extend(int(x) for x in shuffled[n_train + n_val :])

    train.sort()
    val.sort()
    test.sort()
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(path.with_name("training_dataset.h5")),
        "seed": int(seed),
        "rules": {
            "train_fraction": 0.70,
            "val_fraction": 0.15,
            "stratified_by": ["k", "active_ratio_requested"],
        },
        "splits": {
            "train_id": train,
            "val_id": val,
            "test_id": test,
            "train": train,
            "val": val,
            "test": test,
        },
        "counts": {
            "num_scenarios": len(samples),
            "train_id": len(train),
            "val_id": len(val),
            "test_id": len(test),
        },
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_manifest(path: Path, samples: list[dict[str, object]]) -> None:
    fieldnames = [
        "sample_id",
        "sample_index",
        "k",
        "active_ratio_requested",
        "active_ratio_actual",
        "num_active",
        "targets_json",
        "sample_dir",
        "targets_csv",
        "mask_csv",
        "weights_csv",
        "hfss_sources_csv",
        "sample_json",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample["sample_id"],
                    "sample_index": sample["sample_index"],
                    "k": sample["k"],
                    "active_ratio_requested": f"{float(sample['active_ratio_requested']):.3f}",
                    "active_ratio_actual": f"{float(sample['active_ratio_actual']):.9f}",
                    "num_active": sample["num_active"],
                    "targets_json": json.dumps(sample["targets"], separators=(",", ":")),
                    "sample_dir": sample["sample_dir"],
                    "targets_csv": sample["targets_csv"],
                    "mask_csv": sample["mask_csv"],
                    "weights_csv": sample["weights_csv"],
                    "hfss_sources_csv": sample["hfss_sources_csv"],
                    "sample_json": sample["sample_json"],
                }
            )


def write_export_queue(path: Path, samples: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SampleID", "SourcesCSV", "OutputDir"])
        for sample in samples:
            writer.writerow([sample["sample_id"], sample["hfss_sources_csv"], sample["sample_dir"]])


def write_config(path: Path, *, config: GeneratorConfig) -> None:
    payload = asdict(config)
    payload["k_choices"] = list(K_CHOICES)
    payload["active_ratios"] = list(ACTIVE_RATIOS)
    payload["port_naming"] = "P{element_index:03d}"
    payload["element_indexing"] = "ix * ny + iy"
    payload["weight_convention"] = (
        "task_weights use beam-multitask steering exp(+j*k*r_dot_u); "
        "HFSS complex field coefficients use conjugate(sum(task_weights)) normalized to unit L2 norm; "
        "Driven Modal source magnitude is abs(coefficient)^2 W with coefficient phase, so total incident power is 1 W"
    )
    payload["source_convention"] = "driven_modal_incident_power_watt"
    payload["power_normalization"] = "sum(incident_power_w)=1W_per_sample"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_hfss_export_vbs(path: Path, *, out_dir: Path) -> None:
    project_path = ROOT / "models" / "hfss" / "ura16_quick_10ghz_fullarray_run.aedt"
    queue_path = out_dir / "hfss_export_queue.csv"
    content = f'''Option Explicit

Dim projectPath, designName, queuePath, sphereName, solutionName
Dim progressPath, summaryPath
Dim BATCH_SIZE, KEEP_REPORTS, RESUME_ENABLED, FORCE_EXPORT
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSol, oReport, fso
Dim queueFile, header, line, parts, sampleId, sourcesCsv, outputDir
Dim exportedCount, skippedCount, scannedCount

projectPath = "{vbs_path(project_path)}"
designName = "URA16_Quick_10GHz"
queuePath = "{vbs_path(queue_path)}"
sphereName = "InfiniteSphere_Theta0_90_Phi0_360"
solutionName = "Setup_10GHz : LastAdaptive"
BATCH_SIZE = 100
RESUME_ENABLED = True
FORCE_EXPORT = False
KEEP_REPORTS = False

Set fso = CreateObject("Scripting.FileSystemObject")
progressPath = fso.BuildPath(fso.GetParentFolderName(queuePath), "hfss_export_progress.csv")
summaryPath = fso.BuildPath(fso.GetParentFolderName(queuePath), "hfss_export_batch_summary.csv")
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject projectPath
Set oProject = oDesktop.SetActiveProject("ura16_quick_10ghz_fullarray_run")
Set oDesign = oProject.SetActiveDesign(designName)
Set oSol = oDesign.GetModule("Solutions")
Set oReport = oDesign.GetModule("ReportSetup")

Set queueFile = fso.OpenTextFile(queuePath, 1)
If Not queueFile.AtEndOfStream Then header = queueFile.ReadLine
exportedCount = 0
skippedCount = 0
scannedCount = 0

Do Until queueFile.AtEndOfStream
    line = Trim(queueFile.ReadLine)
    If Len(line) > 0 Then
        scannedCount = scannedCount + 1
        parts = Split(line, ",")
        sampleId = parts(0)
        sourcesCsv = parts(1)
        outputDir = parts(2)
        EnsureFolder outputDir
        If RESUME_ENABLED And Not FORCE_EXPORT And SampleComplete(outputDir) Then
            skippedCount = skippedCount + 1
        Else
            ApplySourcesFromCsv oSol, sourcesCsv
            ExportSampleReports oReport, sampleId, outputDir
            If Not SampleComplete(outputDir) Then
                AppendProgress sampleId, "failed_missing_outputs", outputDir
                Err.Raise vbObjectError + 1001, "ExportSampleReports", "Missing HFSS export CSV files for " & sampleId
            End If
            AppendProgress sampleId, "exported", outputDir
            exportedCount = exportedCount + 1
            If BATCH_SIZE > 0 And exportedCount >= BATCH_SIZE Then Exit Do
        End If
    End If
Loop

queueFile.Close
AppendSummary scannedCount, skippedCount, exportedCount, BATCH_SIZE
oProject.Save
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

Sub ApplySourcesFromCsv(solModule, csvPath)
    Dim srcFile, srcHeader, srcLine, srcParts
    Dim magMap, phaseMap, sources, editArgs()
    Dim i, name, mag, phase

    Set magMap = CreateObject("Scripting.Dictionary")
    Set phaseMap = CreateObject("Scripting.Dictionary")

    Set srcFile = fso.OpenTextFile(csvPath, 1)
    If Not srcFile.AtEndOfStream Then srcHeader = srcFile.ReadLine
    Do Until srcFile.AtEndOfStream
        srcLine = Trim(srcFile.ReadLine)
        If Len(srcLine) > 0 Then
            srcParts = Split(srcLine, ",")
            name = srcParts(0)
            magMap(name) = srcParts(1)
            phaseMap(name) = srcParts(2)
        End If
    Loop
    srcFile.Close

    sources = solModule.GetAllSources()
    ReDim editArgs(UBound(sources) + 1)
    editArgs(0) = Array("IncludePortPostProcessing:=", False, "SpecifySystemPower:=", False)

    For i = LBound(sources) To UBound(sources)
        name = CStr(sources(i))
        If magMap.Exists(name) Then
            mag = CStr(magMap(name)) & "W"
            phase = CStr(phaseMap(name)) & "deg"
        Else
            mag = "0W"
            phase = "0deg"
        End If
        editArgs(i + 1) = Array("Name:=", name, "Magnitude:=", mag, "Phase:=", phase)
    Next

    solModule.EditSources editArgs
End Sub

Sub ExportSampleReports(reportModule, sampleId, outputDir)
    Dim rptThetaPhi, rptPhi0, rptPhi90

    rptThetaPhi = "MT_" & sampleId & "_GainTotal_ThetaPhi"
    rptPhi0 = "MT_" & sampleId & "_GainTotal_Phi0"
    rptPhi90 = "MT_" & sampleId & "_GainTotal_Phi90"

    DeleteIfExists reportModule, rptThetaPhi
    DeleteIfExists reportModule, rptPhi0
    DeleteIfExists reportModule, rptPhi90

    reportModule.CreateReport rptThetaPhi, "Far Fields", "Rectangular Contour Plot", solutionName, _
        Array("Context:=", sphereName), _
        Array("Freq:=", Array("10GHz"), "Theta:=", Array("All"), "Phi:=", Array("All")), _
        Array("X Component:=", "Theta", "Y Component:=", "Phi", "Z Component:=", Array("dB(GainTotal)")), _
        Array()
    reportModule.ExportToFile rptThetaPhi, outputDir & "\\hfss_gain_total_theta_phi.csv"

    reportModule.CreateReport rptPhi0, "Far Fields", "Rectangular Plot", solutionName, _
        Array("Context:=", sphereName), _
        Array("Freq:=", Array("10GHz"), "Theta:=", Array("All"), "Phi:=", Array("0deg")), _
        Array("X Component:=", "Theta", "Y Component:=", Array("dB(GainTotal)")), _
        Array()
    reportModule.ExportToFile rptPhi0, outputDir & "\\hfss_gain_total_phi0.csv"

    reportModule.CreateReport rptPhi90, "Far Fields", "Rectangular Plot", solutionName, _
        Array("Context:=", sphereName), _
        Array("Freq:=", Array("10GHz"), "Theta:=", Array("All"), "Phi:=", Array("90deg")), _
        Array("X Component:=", "Theta", "Y Component:=", Array("dB(GainTotal)")), _
        Array()
    reportModule.ExportToFile rptPhi90, outputDir & "\\hfss_gain_total_phi90.csv"

    If Not KEEP_REPORTS Then
        DeleteIfExists reportModule, rptThetaPhi
        DeleteIfExists reportModule, rptPhi0
        DeleteIfExists reportModule, rptPhi90
    End If
End Sub

Sub EnsureFolder(folderPath)
    If Not fso.FolderExists(folderPath) Then
        fso.CreateFolder(folderPath)
    End If
End Sub

Function SampleComplete(outputDir)
    SampleComplete = OutputFileOk(outputDir & "\\hfss_gain_total_theta_phi.csv", 1000) _
        And OutputFileOk(outputDir & "\\hfss_gain_total_phi0.csv", 100) _
        And OutputFileOk(outputDir & "\\hfss_gain_total_phi90.csv", 100)
End Function

Function OutputFileOk(filePath, minBytes)
    OutputFileOk = False
    If fso.FileExists(filePath) Then
        If fso.GetFile(filePath).Size >= minBytes Then
            OutputFileOk = True
        End If
    End If
End Function

Sub AppendProgress(sampleId, status, outputDir)
    Dim progressFile
    Set progressFile = OpenCsvAppend(progressPath, "timestamp,sample_id,status,output_dir")
    progressFile.WriteLine CsvCell(TimestampNow()) & "," & CsvCell(sampleId) & "," & CsvCell(status) & "," & CsvCell(outputDir)
    progressFile.Close
End Sub

Sub AppendSummary(scannedCount, skippedCount, exportedCount, batchSize)
    Dim summaryFile
    Set summaryFile = OpenCsvAppend(summaryPath, "timestamp,scanned_rows,skipped_existing,exported_new,batch_size,resume_enabled,force_export")
    summaryFile.WriteLine CsvCell(TimestampNow()) & "," & scannedCount & "," & skippedCount & "," & exportedCount & "," & batchSize & "," & RESUME_ENABLED & "," & FORCE_EXPORT
    summaryFile.Close
End Sub

Function OpenCsvAppend(filePath, headerLine)
    Dim shouldWriteHeader, csvFile
    shouldWriteHeader = True
    If fso.FileExists(filePath) Then
        If fso.GetFile(filePath).Size > 0 Then shouldWriteHeader = False
    End If
    Set csvFile = fso.OpenTextFile(filePath, 8, True)
    If shouldWriteHeader Then csvFile.WriteLine headerLine
    Set OpenCsvAppend = csvFile
End Function

Function CsvCell(value)
    CsvCell = """" & Replace(CStr(value), """", """""") & """"
End Function

Function TimestampNow()
    Dim d
    d = Now
    TimestampNow = Year(d) & "-" & Pad2(Month(d)) & "-" & Pad2(Day(d)) & " " & Pad2(Hour(d)) & ":" & Pad2(Minute(d)) & ":" & Pad2(Second(d))
End Function

Function Pad2(value)
    Pad2 = Right("0" & CStr(value), 2)
End Function

Sub DeleteIfExists(reportModule, name)
    On Error Resume Next
    reportModule.DeleteReports Array(name)
    On Error GoTo 0
End Sub
'''
    path.write_text(content, encoding="ascii")


def write_hfss_export_ps1(path: Path, *, out_dir: Path) -> None:
    script_path = out_dir / "export_hfss_patterns.vbs"
    log_path = out_dir / "export_hfss_patterns.log"
    queue_path = out_dir / "hfss_export_queue.csv"
    content = f'''$ErrorActionPreference = "Stop"

$ansys = "D:\\v231\\Win64\\ansysedt.exe"
$script = "{ps_path(script_path)}"
$log = "{ps_path(log_path)}"
$queue = "{ps_path(queue_path)}"

function Get-HfssExportStatus {{
    param([string]$QueuePath)

    $rows = Import-Csv -LiteralPath $QueuePath
    $done = 0
    foreach ($row in $rows) {{
        $thetaPhi = Join-Path $row.OutputDir "hfss_gain_total_theta_phi.csv"
        $phi0 = Join-Path $row.OutputDir "hfss_gain_total_phi0.csv"
        $phi90 = Join-Path $row.OutputDir "hfss_gain_total_phi90.csv"
        if ((Test-Path -LiteralPath $thetaPhi) -and
            (Test-Path -LiteralPath $phi0) -and
            (Test-Path -LiteralPath $phi90) -and
            ((Get-Item -LiteralPath $thetaPhi).Length -ge 1000) -and
            ((Get-Item -LiteralPath $phi0).Length -ge 100) -and
            ((Get-Item -LiteralPath $phi90).Length -ge 100)) {{
            $done += 1
        }}
    }}

    [pscustomobject]@{{
        Done = $done
        Total = $rows.Count
        Remaining = $rows.Count - $done
    }}
}}

if (-not (Test-Path -LiteralPath $ansys)) {{
    throw "ansysedt.exe not found: $ansys"
}}
if (-not (Test-Path -LiteralPath $script)) {{
    throw "HFSS export script not found: $script"
}}
if (-not (Test-Path -LiteralPath $queue)) {{
    throw "HFSS export queue not found: $queue"
}}

Remove-Item -LiteralPath $log -Force -ErrorAction SilentlyContinue
$before = Get-HfssExportStatus -QueuePath $queue
Write-Host "HFSS export status before run: $($before.Done)/$($before.Total) completed, $($before.Remaining) remaining."
Write-Host "This run exports up to the next 100 missing samples."

$proc = Start-Process -FilePath $ansys -ArgumentList @("-RunScriptAndExit", $script) -PassThru -WindowStyle Hidden
$deadline = (Get-Date).AddHours(8)

while (-not $proc.HasExited) {{
    if ((Get-Date) -gt $deadline) {{
        throw "Timed out waiting for HFSS export."
    }}
    Start-Sleep -Seconds 10
    $proc.Refresh()
}}

if ($proc.ExitCode -ne 0) {{
    throw "HFSS export exited with code $($proc.ExitCode)."
}}

$after = Get-HfssExportStatus -QueuePath $queue
Write-Host "HFSS pattern export completed."
Write-Host "HFSS export status after run: $($after.Done)/$($after.Total) completed, $($after.Remaining) remaining."
'''
    path.write_text(content, encoding="ascii")


def write_readme(path: Path, *, out_dir: Path, samples: list[dict[str, object]], config: GeneratorConfig) -> None:
    content = f"""# Multi-task HFSS Dataset

Generated samples: {len(samples)}

Array:
- URA16, 16 x 16 elements
- dx = dy = 0.5 lambda
- element index = ix * ny + iy
- HFSS port name = P000 ... P255

Sampling:
- K choices: {list(K_CHOICES)}
- active ratios: {list(ACTIVE_RATIOS)}
- theta range: [{config.theta_min_deg}, {config.theta_max_deg}] deg
- phi range: [{config.phi_min_deg}, {config.phi_max_deg}) deg
- minimum target separation: {config.min_separation_deg} deg

Files:
- manifest.csv: one row per generated sample
- dataset_arrays.npz: compact ML-friendly arrays
- samples/sample_xxxxxx/targets.csv: target angles
- samples/sample_xxxxxx/mask.csv: active element mask
- samples/sample_xxxxxx/weights.csv: mask + per-task weights + conjugated HFSS combined excitation
- samples/sample_xxxxxx/hfss_sources.csv: direct HFSS port magnitude/phase table
- hfss_export_queue.csv: queue consumed by export_hfss_patterns.vbs
- export_hfss_patterns.vbs: AEDT/HFSS post-processing script
- run_hfss_export_patterns.ps1: one-click HFSS export wrapper
- hfss_export_progress.csv: appended per-sample HFSS export progress, created after export
- hfss_export_batch_summary.csv: appended per-batch summary, created after export

Usage:

```powershell
python "{ps_path(ROOT / "scripts" / "generate_multitask_hfss_dataset.py")}" --samples-per-combo 10
```

To export HFSS patterns in resumable batches of 100 missing samples per run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "{ps_path(out_dir / "run_hfss_export_patterns.ps1")}"
```

The script skips samples whose three HFSS output CSV files already exist and are non-empty:

- hfss_gain_total_theta_phi.csv
- hfss_gain_total_phi0.csv
- hfss_gain_total_phi90.csv

To change the batch size, edit `export_hfss_patterns.vbs`:

```vbscript
BATCH_SIZE = 100
```

The HFSS export script reuses the solved project:

```text
{ROOT / "models" / "hfss" / "ura16_quick_10ghz_fullarray_run.aedt"}
```

If this project has no solved field data, run the full-array solve first.
"""
    path.write_text(content, encoding="utf-8")


def phase_deg(value: complex) -> float:
    if abs(value) <= 0.0:
        return 0.0
    phase = math.degrees(math.atan2(value.imag, value.real))
    while phase >= 180.0:
        phase -= 360.0
    while phase < -180.0:
        phase += 360.0
    return phase


def vbs_path(path: Path) -> str:
    return str(path).replace('"', '""')


def ps_path(path: Path) -> str:
    return str(path).replace('"', '`"')


if __name__ == "__main__":
    main()
