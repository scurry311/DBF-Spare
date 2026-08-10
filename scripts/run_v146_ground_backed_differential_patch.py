#!/usr/bin/env python3
"""Build and gate a true differential split patch over a continuous ground."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_v144_grounded_cavity_patch as pipeline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v146_ground_backed_differential_patch_preregistered.json"
DESIGN_NAME = "V146_GroundBackedDifferentialPatch"


def differential_geometry_text(
    side: int, geometry: dict[str, Any]
) -> tuple[str, list[str], list[str]]:
    sx = float(geometry["spacing_x_mm"])
    sy = float(geometry["spacing_y_mm"])
    if abs(sx - sy) > 1.0e-9:
        raise ValueError("v1.46 currently requires square cells")
    board_x, board_y = side * sx, side * sy
    h = float(geometry["substrate_thickness_mm"])
    copper = float(geometry["copper_thickness_mm"])
    patch_x = float(geometry["patch_width_x_mm"])
    patch_y = float(geometry["patch_total_length_y_mm"])
    gap = float(geometry["split_gap_y_mm"])
    feed_sep = float(geometry["differential_probe_spacing_y_mm"])
    probe = float(geometry["probe_radius_mm"])
    clearance = float(geometry["ground_clearance_radius_mm"])
    drop = float(geometry["probe_drop_below_ground_mm"])
    pad_x = float(geometry["bottom_pad_width_x_mm"])
    pad_y = float(geometry["bottom_pad_length_y_mm"])

    arm_y = (patch_y - gap) / 2.0
    if arm_y <= 0.5 or patch_x >= sx - 0.6 or patch_y >= sy - 0.6:
        raise ValueError("Invalid split-patch envelope")
    if feed_sep / 2.0 - probe <= gap / 2.0:
        raise ValueError("Differential probe overlaps the split gap")
    if feed_sep / 2.0 + probe >= patch_y / 2.0:
        raise ValueError("Differential probe lies outside the patch")
    if 2.0 * clearance >= feed_sep:
        raise ValueError("Ground-clearance holes overlap")
    if pad_y >= feed_sep:
        raise ValueError("Differential bottom pads overlap")

    lines: list[str] = [
        f'CreateBox oEditor, "Substrate", {-board_x/2:.7f}, {-board_y/2:.7f}, '
        f'0, {board_x:.7f}, {board_y:.7f}, {h:.7f}, "RO5880_V144", True',
        f'CreateSheetZ oEditor, "Ground", {-board_x/2:.7f}, {-board_y/2:.7f}, '
        f'0, {board_x:.7f}, {board_y:.7f}',
    ]
    sheet_conductors = ["Ground"]
    mesh_names: list[str] = []
    port_names: list[str] = []

    for name, cx, cy in pipeline.centers(side, sx):
        yn = cy - feed_sep / 2.0
        yp = cy + feed_sep / 2.0
        lower = f"PatchN_{name}"
        upper = f"PatchP_{name}"
        pad_n = f"PadN_{name}"
        pad_p = f"PadP_{name}"
        port = f"PortSheet_{name}"
        inner_n = yn + pad_y / 2.0
        inner_p = yp - pad_y / 2.0

        lines.extend(
            [
                f'CreateSheetZ oEditor, "{lower}", {cx-patch_x/2:.7f}, '
                f'{cy-gap/2-arm_y:.7f}, {h:.7f}, {patch_x:.7f}, {arm_y:.7f}',
                f'CreateSheetZ oEditor, "{upper}", {cx-patch_x/2:.7f}, '
                f'{cy+gap/2:.7f}, {h:.7f}, {patch_x:.7f}, {arm_y:.7f}',
                f'CreateCylinderZ oEditor, "GroundHoleN_{name}", {cx:.7f}, {yn:.7f}, '
                f'-0.01, {clearance:.7f}, 0.02, "vacuum", True',
                f'SubtractObject oEditor, "Ground", "GroundHoleN_{name}"',
                f'CreateCylinderZ oEditor, "GroundHoleP_{name}", {cx:.7f}, {yp:.7f}, '
                f'-0.01, {clearance:.7f}, 0.02, "vacuum", True',
                f'SubtractObject oEditor, "Ground", "GroundHoleP_{name}"',
                f'CreateCylinderZ oEditor, "SubstrateHoleN_{name}", {cx:.7f}, {yn:.7f}, '
                f'-0.01, {probe+0.02:.7f}, {h+0.02:.7f}, "vacuum", True',
                f'SubtractObject oEditor, "Substrate", "SubstrateHoleN_{name}"',
                f'CreateCylinderZ oEditor, "SubstrateHoleP_{name}", {cx:.7f}, {yp:.7f}, '
                f'-0.01, {probe+0.02:.7f}, {h+0.02:.7f}, "vacuum", True',
                f'SubtractObject oEditor, "Substrate", "SubstrateHoleP_{name}"',
                f'CreateCylinderZ oEditor, "ProbeN_{name}", {cx:.7f}, {yn:.7f}, '
                f'{-drop:.7f}, {probe:.7f}, {h+drop:.7f}, "copper", False',
                f'CreateCylinderZ oEditor, "ProbeP_{name}", {cx:.7f}, {yp:.7f}, '
                f'{-drop:.7f}, {probe:.7f}, {h+drop:.7f}, "copper", False',
                f'CreateSheetZ oEditor, "{pad_n}", {cx-pad_x/2:.7f}, '
                f'{yn-pad_y/2:.7f}, {-drop:.7f}, {pad_x:.7f}, {pad_y:.7f}',
                f'CreateSheetZ oEditor, "{pad_p}", {cx-pad_x/2:.7f}, '
                f'{yp-pad_y/2:.7f}, {-drop:.7f}, {pad_x:.7f}, {pad_y:.7f}',
                f'CreateSheetZ oEditor, "{port}", {cx-pad_x/2:.7f}, '
                f'{inner_n:.7f}, {-drop:.7f}, {pad_x:.7f}, {inner_p-inner_n:.7f}',
                f'AssignPort oBoundary, "{name}", "{port}", '
                f'{cx:.7f}, {inner_n+0.01:.7f}, {-drop:.7f}, '
                f'{cx:.7f}, {inner_p-0.01:.7f}, {-drop:.7f}',
            ]
        )
        sheet_conductors.extend([lower, upper, pad_n, pad_p])
        mesh_names.append(port)
        port_names.append(name)

    finite_array = ", ".join(f'"{name}"' for name in sheet_conductors)
    lines.append(
        'oBoundary.AssignFiniteCond Array("NAME:CopperSheetFiniteConductivity", '
        f'"Objects:=", Array({finite_array}), "UseMaterial:=", True, '
        '"Material:=", "copper", "UseThickness:=", True, '
        f'"Thickness:=", "{copper:.7f}mm", "Roughness:=", "0um", '
        '"InfGroundPlane:=", False, "IsTwoSided:=", True, '
        '"IsShellElement:=", False)'
    )
    return "\n".join(lines), mesh_names, port_names


_prepare_case = pipeline.prepare_case
_preregister = pipeline.preregister
_analyze_1x1 = pipeline.analyze_1x1


def prepare_case_v146(
    root: Path,
    case_id: str,
    side: int,
    geometry: dict[str, Any],
    frequency_ghz: float,
    solver_type: str,
) -> dict[str, Any]:
    case = _prepare_case(root, case_id, side, geometry, frequency_ghz, solver_type)
    folder = Path(case["project_path"]).parent
    project = folder / f"v146_{case_id}.aedt"
    touchstone = folder / f"v146_{case_id}.s{side * side}p"
    builder = folder / "build.vbs"
    solver = folder / "solve_export.vbs"
    port_names = [
        name for name, _, _ in pipeline.centers(side, float(geometry["spacing_x_mm"]))
    ]
    builder.write_text(
        pipeline.builder_text(project, side, geometry, frequency_ghz, solver_type),
        encoding="ascii",
    )
    solver.write_text(
        pipeline.solver_text(project, touchstone, folder, port_names, frequency_ghz),
        encoding="ascii",
    )
    case.update(
        {
            "project_path": str(project.resolve()),
            "touchstone_path": str(touchstone.resolve()),
            "builder_sha256": pipeline.sha256(builder),
            "solver_sha256": pipeline.sha256(solver),
            "cad_audit": {
                "radiator_type": "true_differential_split_patch",
                "continuous_reflecting_ground": True,
                "external_port_count": side * side,
                "differential_probe_count": 2 * side * side,
                "ground_clearance_hole_count": 2 * side * side,
                "finite_conductivity_sheet_model": True,
                "external_matching_stage_count": 0,
                "input_controls": [
                    "patch_total_length_y_mm",
                    "split_gap_y_mm",
                    "differential_probe_spacing_y_mm",
                ],
            },
        }
    )
    pipeline.write_json(folder / "case_manifest.json", case)
    return case


def preregister_v146(config: dict[str, Any]) -> dict[str, Any]:
    result = _preregister(config)
    out = pipeline.output_root(config)
    decision = pipeline.read_json(out / "stage_decision.json")
    decision.update(
        {
            "stage": "A_v146_ground_backed_differential_preregistered",
            "reason": (
                "The one-port true-differential split patch over a continuous "
                "reflecting ground is preregistered; only 1x1 is open."
            ),
        }
    )
    pipeline.write_json(out / "stage_decision.json", decision)
    result["decision"] = decision
    return result


def analyze_1x1_v146(config: dict[str, Any]) -> dict[str, Any]:
    result = _analyze_1x1(config)
    out = pipeline.output_root(config)
    decision = pipeline.read_json(out / "stage_decision.json")
    decision["reason"] = (
        "A ground-backed true-differential element passed the frozen 1x1 gate; one independent repeat is required before 2x2."
        if decision.get("allow_2x2")
        else "No ground-backed true-differential element passed the 15 dB 1x1 gate; 2x2 remains locked."
    )
    pipeline.write_json(out / "stage_decision.json", decision)
    result["decision"] = decision
    return result


def main() -> None:
    pipeline.DEFAULT_CONFIG = DEFAULT_CONFIG
    pipeline.DESIGN_NAME = DESIGN_NAME
    pipeline.cavity_geometry_text = differential_geometry_text
    pipeline.prepare_case = prepare_case_v146
    pipeline.preregister = preregister_v146
    pipeline.analyze_1x1 = analyze_1x1_v146
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=("preregister", "prepare-1x1", "run-1x1", "analyze-1x1", "prepare-2x2", "run-2x2", "analyze-2x2", "status"),
        default="status",
    )
    args = parser.parse_args()
    config = pipeline.load_config(args.config)
    actions = {
        "preregister": preregister_v146,
        "prepare-1x1": pipeline.prepare_1x1,
        "run-1x1": pipeline.run_1x1,
        "analyze-1x1": analyze_1x1_v146,
        "prepare-2x2": pipeline.prepare_2x2,
        "run-2x2": pipeline.run_2x2,
        "analyze-2x2": pipeline.analyze_2x2,
        "status": pipeline.status,
    }
    print(json.dumps(actions[args.mode](config), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
