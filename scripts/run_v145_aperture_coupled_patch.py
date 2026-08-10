#!/usr/bin/env python3
"""Build and gate a standard aperture-coupled patch at 1x1 and 2x2.

This branch deliberately removes the v1.27 slotted radiator and coplanar GSG
launch.  A complete patch is coupled through one transverse slot in a
continuous ground plane.  Patch length, slot length, and open-stub length are
independent controls for resonance, coupling, and input reactance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_v144_grounded_cavity_patch as pipeline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v145_aperture_coupled_patch_preregistered.json"
DESIGN_NAME = "V145_ApertureCoupledPatch"


def aperture_geometry_text(
    side: int, geometry: dict[str, Any]
) -> tuple[str, list[str], list[str]]:
    spacing_x = float(geometry["spacing_x_mm"])
    spacing_y = float(geometry["spacing_y_mm"])
    if abs(spacing_x - spacing_y) > 1.0e-9:
        raise ValueError("v1.45 currently requires square cells")

    top_h = float(geometry["top_substrate_thickness_mm"])
    feed_h = float(geometry["feed_substrate_thickness_mm"])
    copper = float(geometry["copper_thickness_mm"])
    patch_x = float(geometry["patch_width_x_mm"])
    patch_y = float(geometry["patch_length_y_mm"])
    slot_l = float(geometry["aperture_length_x_mm"])
    slot_w = float(geometry["aperture_width_y_mm"])
    slot_y = float(geometry.get("aperture_offset_y_mm", 0.0))
    line_w = float(geometry["feed_line_width_mm"])
    stub_l = float(geometry["open_stub_length_mm"])
    transformer_enabled = bool(geometry.get("transformer_enabled", False))
    transformer_width = float(geometry.get("transformer_width_mm", line_w))
    transformer_length = float(geometry.get("transformer_length_mm", 0.0))
    transformer_gap = float(
        geometry.get("transformer_to_aperture_gap_mm", 0.0)
    )
    via_enabled = bool(geometry.get("grounded_via_fence_enabled", True))
    via_radius = float(geometry.get("grounded_via_radius_mm", 0.15))
    via_pitch = float(geometry.get("grounded_via_pitch_mm", 2.5))
    via_edge_inset = float(geometry.get("grounded_via_edge_inset_mm", 0.35))

    if patch_x >= spacing_x - 0.6 or patch_y >= spacing_y - 0.6:
        raise ValueError("Patch does not clear the cell boundary")
    if slot_l >= spacing_x - 0.6 or slot_w <= 0.0:
        raise ValueError("Invalid coupling-aperture dimensions")
    if line_w >= spacing_x / 2.0:
        raise ValueError("Feed line is too wide for the cell")
    if transformer_enabled and transformer_width >= spacing_x - 0.6:
        raise ValueError("Transformer does not clear the cell boundary")
    if stub_l >= spacing_y / 2.0 - 0.3:
        raise ValueError("Open stub crosses the next local port plane")

    board_x = side * spacing_x
    board_y = side * spacing_y
    ground_z = 0.0
    top_z = ground_z
    patch_z = top_z + top_h
    feed_z = -feed_h

    lines: list[str] = [
        f'CreateBox oEditor, "TopSubstrate", {-board_x/2:.7f}, {-board_y/2:.7f}, '
        f'{top_z:.7f}, {board_x:.7f}, {board_y:.7f}, {top_h:.7f}, "RO5880_V144", True',
        f'CreateSheetZ oEditor, "Ground", {-board_x/2:.7f}, {-board_y/2:.7f}, '
        f'{ground_z:.7f}, {board_x:.7f}, {board_y:.7f}',
        f'CreateBox oEditor, "FeedSubstrate", {-board_x/2:.7f}, {-board_y/2:.7f}, '
        f'{-feed_h:.7f}, {board_x:.7f}, {board_y:.7f}, {feed_h:.7f}, "RO5880_V144", True',
    ]
    mesh_names: list[str] = []
    conductor_names: list[str] = []
    port_names: list[str] = []

    for name, cx, cy in pipeline.centers(side, spacing_x):
        patch = f"Patch_{name}"
        aperture = f"Aperture_{name}"
        feed = f"FeedLine_{name}"
        port = f"PortSheet_{name}"
        aperture_y = cy + slot_y
        line_start = cy - spacing_y / 2.0
        line_stop = aperture_y + stub_l
        line_length = line_stop - line_start
        if line_length <= 0.5:
            raise ValueError(f"Non-positive feed length for {name}")

        feed_geometry: list[str]
        if transformer_enabled:
            transformer_stop = aperture_y - transformer_gap
            transformer_start = transformer_stop - transformer_length
            if transformer_length <= 0.2 or transformer_start <= line_start + 0.2:
                raise ValueError("Transformer length leaves no 50-ohm input line")
            transition_geometry = []
            transition_name = ""
            if transformer_gap > 1.0e-9:
                transition_geometry = [
                    f'CreateSheetZ oEditor, "FeedTransition_{name}", {cx-line_w/2:.7f}, '
                    f'{transformer_stop:.7f}, {feed_z:.7f}, {line_w:.7f}, {transformer_gap:.7f}'
                ]
                transition_name = f",FeedTransition_{name}"
            feed_geometry = [
                f'CreateSheetZ oEditor, "{feed}", {cx-line_w/2:.7f}, '
                f'{line_start:.7f}, {feed_z:.7f}, {line_w:.7f}, '
                f'{transformer_start-line_start:.7f}',
                f'CreateSheetZ oEditor, "FeedTransformer_{name}", '
                f'{cx-transformer_width/2:.7f}, {transformer_start:.7f}, '
                f'{feed_z:.7f}, {transformer_width:.7f}, {transformer_length:.7f}',
                *transition_geometry,
                f'CreateSheetZ oEditor, "FeedStub_{name}", {cx-line_w/2:.7f}, '
                f'{aperture_y:.7f}, {feed_z:.7f}, {line_w:.7f}, {stub_l:.7f}',
                f'UniteSelection oEditor, "{feed},FeedTransformer_{name}{transition_name},FeedStub_{name}"',
            ]
        else:
            feed_geometry = [
                f'CreateSheetZ oEditor, "{feed}", {cx-line_w/2:.7f}, '
                f'{line_start:.7f}, {feed_z:.7f}, {line_w:.7f}, {line_length:.7f}'
            ]

        lines.extend(
            [
                f'CreateSheetZ oEditor, "{patch}", {cx-patch_x/2:.7f}, '
                f'{cy-patch_y/2:.7f}, {patch_z:.7f}, {patch_x:.7f}, {patch_y:.7f}',
                f'CreateSheetZ oEditor, "{aperture}", {cx-slot_l/2:.7f}, '
                f'{aperture_y-slot_w/2:.7f}, {ground_z:.7f}, {slot_l:.7f}, {slot_w:.7f}',
                f'SubtractObject oEditor, "Ground", "{aperture}"',
                *feed_geometry,
                f'CreateSheetY oEditor, "{port}", {cx-line_w/2:.7f}, '
                f'{line_start:.7f}, {-feed_h:.7f}, {line_w:.7f}, '
                f'{feed_h:.7f}',
                f'AssignPort oBoundary, "{name}", "{port}", '
                f'{cx:.7f}, {line_start:.7f}, {-feed_h+0.01:.7f}, '
                f'{cx:.7f}, {line_start:.7f}, {-0.01:.7f}',
            ]
        )
        conductor_names.extend([patch, feed])
        mesh_names.append(port)
        port_names.append(name)

    if via_enabled:
        via_names: list[str] = []
        seen: set[tuple[int, int]] = set()

        def add_via(x: float, y: float) -> None:
            key = (round(x * 10000), round(y * 10000))
            if key in seen:
                return
            seen.add(key)
            name = f"GroundVia_{len(via_names):03d}"
            via_names.append(name)
            lines.extend(
                [
                    f'CreateCylinderZ oEditor, "{name}", {x:.7f}, {y:.7f}, '
                    f'{ground_z:.7f}, {via_radius:.7f}, {top_h:.7f}, "copper", False',
                    f'CreateCylinderZ oEditor, "ViaCut_{name}", {x:.7f}, {y:.7f}, '
                    f'{top_z-0.01:.7f}, {via_radius+0.01:.7f}, {top_h+0.02:.7f}, "vacuum", True',
                    f'SubtractObject oEditor, "TopSubstrate", "ViaCut_{name}"',
                ]
            )

        x_boundaries = [
            -board_x / 2.0 + via_edge_inset,
            *[-board_x / 2.0 + i * spacing_x for i in range(1, side)],
            board_x / 2.0 - via_edge_inset,
        ]
        y_boundaries = [
            -board_y / 2.0 + via_edge_inset,
            *[-board_y / 2.0 + i * spacing_y for i in range(1, side)],
            board_y / 2.0 - via_edge_inset,
        ]
        ny = max(1, round(board_y / via_pitch))
        nx = max(1, round(board_x / via_pitch))
        for x in x_boundaries:
            for index in range(ny + 1):
                add_via(x, -board_y / 2.0 + index * board_y / ny)
        for y in y_boundaries:
            for index in range(nx + 1):
                add_via(-board_x / 2.0 + index * board_x / nx, y)
        if via_names:
            # The posts are volumetric copper touching the finite-conductivity
            # ground sheet.  They intentionally remain separate CAD bodies.
            pass

    finite_objects = ["Ground", *conductor_names]
    finite_array = ", ".join(f'"{name}"' for name in finite_objects)
    lines.append(
        'oBoundary.AssignFiniteCond Array("NAME:CopperSheetFiniteConductivity", '
        f'"Objects:=", Array({finite_array}), "UseMaterial:=", True, '
        '"Material:=", "copper", "UseThickness:=", True, '
        f'"Thickness:=", "{copper:.7f}mm", "Roughness:=", "0um", '
        '"InfGroundPlane:=", False, "IsTwoSided:=", True, '
        '"IsShellElement:=", False)'
    )

    return "\n".join(lines), mesh_names, port_names


_original_prepare_case = pipeline.prepare_case
_original_preregister = pipeline.preregister
_original_analyze_1x1 = pipeline.analyze_1x1


def prepare_case_v145(
    root: Path,
    case_id: str,
    side: int,
    geometry: dict[str, Any],
    frequency_ghz: float,
    solver_type: str,
) -> dict[str, Any]:
    case = _original_prepare_case(
        root, case_id, side, geometry, frequency_ghz, solver_type
    )
    folder = Path(case["project_path"]).parent
    old_project = Path(case["project_path"])
    old_touchstone = Path(case["touchstone_path"])
    project = folder / f"v145_{case_id}.aedt"
    touchstone = folder / f"v145_{case_id}.s{side * side}p"
    builder = folder / "build.vbs"
    solver = folder / "solve_export.vbs"

    builder.write_text(
        pipeline.builder_text(project, side, geometry, frequency_ghz, solver_type),
        encoding="ascii",
    )
    port_names = [
        name for name, _, _ in pipeline.centers(side, float(geometry["spacing_x_mm"]))
    ]
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
                "radiator_type": "complete_rectangular_patch",
                "continuous_reflecting_ground": True,
                "ground_aperture_count": side * side,
                "microstrip_open_stub_count": side * side,
                "vertical_ground_referenced_port_count": side * side,
                "finite_conductivity_sheet_model": True,
                "integrated_single_section_transformer_count": (
                    side * side if geometry.get("transformer_enabled") else 0
                ),
                "input_impedance_controls": [
                    "patch_length_y_mm",
                    "aperture_length_x_mm",
                    "open_stub_length_mm",
                ],
                "external_matching_stage_count": 0,
            },
        }
    )
    if old_project != project and old_project.exists():
        old_project.unlink()
    if old_touchstone != touchstone and old_touchstone.exists():
        old_touchstone.unlink()
    pipeline.write_json(folder / "case_manifest.json", case)
    return case


def preregister_v145(config: dict[str, Any]) -> dict[str, Any]:
    result = _original_preregister(config)
    out = pipeline.output_root(config)
    decision = pipeline.read_json(out / "stage_decision.json")
    decision.update(
        {
            "stage": "A_v145_aperture_patch_preregistered",
            "reason": (
                "The standard aperture-coupled patch branch is preregistered; "
                "only the physical 1x1 resonance/coupling/stub prescreen is open."
            ),
        }
    )
    pipeline.write_json(out / "stage_decision.json", decision)
    result["decision"] = decision
    return result


def analyze_1x1_v145(config: dict[str, Any]) -> dict[str, Any]:
    result = _original_analyze_1x1(config)
    out = pipeline.output_root(config)
    decision = pipeline.read_json(out / "stage_decision.json")
    if decision.get("allow_2x2"):
        decision["reason"] = (
            "A standard aperture-coupled element passed the frozen 1x1 "
            "impedance, efficiency, convergence, and topology gates; one 2x2 "
            "S4 smoke is authorized."
        )
    else:
        decision["reason"] = (
            "No standard aperture-coupled element passed the 15 dB physical "
            "1x1 gate; 2x2 remains locked pending a sensitivity-guided v1.45 run."
        )
    pipeline.write_json(out / "stage_decision.json", decision)
    result["decision"] = decision
    return result


def main() -> None:
    pipeline.DEFAULT_CONFIG = DEFAULT_CONFIG
    pipeline.DESIGN_NAME = DESIGN_NAME
    pipeline.cavity_geometry_text = aperture_geometry_text
    pipeline.prepare_case = prepare_case_v145
    pipeline.preregister = preregister_v145
    pipeline.analyze_1x1 = analyze_1x1_v145

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=(
            "preregister",
            "prepare-1x1",
            "run-1x1",
            "analyze-1x1",
            "prepare-2x2",
            "run-2x2",
            "analyze-2x2",
            "status",
        ),
        default="status",
    )
    args = parser.parse_args()
    config = pipeline.load_config(args.config)
    actions = {
        "preregister": preregister_v145,
        "prepare-1x1": pipeline.prepare_1x1,
        "run-1x1": pipeline.run_1x1,
        "analyze-1x1": analyze_1x1_v145,
        "prepare-2x2": pipeline.prepare_2x2,
        "run-2x2": pipeline.run_2x2,
        "analyze-2x2": pipeline.analyze_2x2,
        "status": pipeline.status,
    }
    print(json.dumps(actions[args.mode](config), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
