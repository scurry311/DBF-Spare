#!/usr/bin/env python3
"""Build and gate the v1.50 truncated-coax wave-port element.

The workflow is evidence-gated. Periodic, finite-array, label-generation, and
critic stages cannot be opened by a surrogate or an incomplete HFSS export.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from contextlib import contextmanager

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "v150_truncated_coax_wave_port_preregistered.json"
)
V149_CONFIG = (
    ROOT / "configs" / "v149_siw_cavity_stacked_patch_preregistered.json"
)
FROZEN_RESOURCES = {
    "minimum_free_memory_before_build_gib": 8.0,
    "minimum_free_memory_before_solve_gib": 13.0,
    "abort_free_memory_during_solve_gib": 3.0,
    "poll_interval_seconds": 5,
    "maximum_concurrent_aedt_instances": 1,
}
FROZEN_GATES = {
    "required_periodic_scan_state_count": 45,
    "minimum_periodic_active_rl_db": 13.0,
    "minimum_broadside_passive_rl_db": 15.0,
    "preferred_broadside_passive_rl_db": 18.0,
    "minimum_periodic_efficiency": 0.97,
    "maximum_scan_gain_drop_db": 3.0,
    "maximum_final_delta_s": 0.05,
    "maximum_critical_warning_count": 0,
    "minimum_2x2_passive_rl_db": 12.0,
    "minimum_active_rl_db": 11.0,
    "minimum_total_rl_db": 11.0,
    "minimum_system_efficiency": 0.95,
    "preferred_maximum_neighbor_coupling_db": -18.0,
    "maximum_direct_ddm_abs_delta_s": 0.05,
    "minimum_4x4_active_rl_db": 11.0,
    "minimum_4x4_total_rl_db": 11.0,
    "minimum_frozen_scene_overall_pass_count": 18,
    "frozen_scene_count": 20,
    "minimum_each_k_pass_rate": 0.8,
}
FROZEN_FREQUENCIES_GHZ = [9.96, 10.0, 10.04]
FROZEN_SWEEP_GHZ = {"start": 9.8, "stop": 10.2, "step": 0.01}
FROZEN_SCAN = {
    "theta_deg": [0.0, 15.0, 30.0, 40.0, 48.0],
    "phi_deg": [0.0, 45.0, 90.0],
}
FROZEN_V149_GEOMETRY_FIELDS = (
    "period_x_mm",
    "period_y_mm",
    "driven_patch_length_mm",
    "driven_patch_width_mm",
    "stacked_patch_length_mm",
    "stacked_patch_width_mm",
    "main_substrate_thickness_mm",
    "stack_spacer_thickness_mm",
    "feed_offset_x_mm",
    "feed_offset_y_mm",
    "probe_radius_mm",
    "siw_via_diameter_mm",
    "siw_via_pitch_mm",
    "siw_via_inset_mm",
    "siw_top_aperture_clearance_mm",
    "local_mesh_probe_mm",
    "local_mesh_patch_edge_mm",
    "local_mesh_via_mm",
    "adaptive_refinement_percent",
    "maximum_passes",
)
FROZEN_WAVE_PORT = {
    "type": "coaxial_modal_wave_port",
    "reference_plane_z_mm": -1.0,
    "mode_count": 1,
    "renormalization_ohm": 50.0,
    "deembed": False,
}


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def load_run_config(run_root: Path) -> dict[str, Any]:
    preregistration = run_root / "preregistration.json"
    audit_path = run_root / "baseline_audit.json"
    if not preregistration.exists() or not audit_path.exists():
        raise FileNotFoundError(
            "Run is missing preregistration.json or baseline_audit.json"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_hash = audit.get("preregistration_sha256")
    actual_hash = sha256(preregistration)
    if expected_hash != actual_hash:
        raise RuntimeError(
            "Run preregistration hash changed after allocation"
        )
    config = json.loads(preregistration.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="ascii")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=check,
    )
    return result.stdout.strip()


def scan_states(config: dict[str, Any]) -> list[dict[str, float]]:
    return [
        {
            "frequency_ghz": float(frequency),
            "theta_deg": float(theta),
            "phi_deg": float(phi),
        }
        for frequency in config["frequencies_ghz"]
        for theta in config["scan"]["theta_deg"]
        for phi in config["scan"]["phi_deg"]
    ]


def validate_config(config: dict[str, Any]) -> None:
    geometry = config["nominal_geometry"]
    if config.get("protocol") != "v1.50-truncated-coax-wave-port":
        raise ValueError("Unexpected v1.50 protocol")
    if "port_definition" in config:
        raise ValueError("v1.50 forbids legacy lumped-port fields")
    if config.get("wave_port") != FROZEN_WAVE_PORT:
        raise ValueError("v1.50 wave-port definition is immutable")
    v149_geometry = json.loads(V149_CONFIG.read_text(encoding="utf-8"))[
        "nominal_geometry"
    ]
    for name in FROZEN_V149_GEOMETRY_FIELDS:
        if geometry.get(name) != v149_geometry.get(name):
            raise ValueError(f"Changed frozen v1.49 geometry field: {name}")
    if float(geometry["period_x_mm"]) != 15.0:
        raise ValueError("v1.50 requires period_x_mm = 15.0")
    if float(geometry["period_y_mm"]) != 15.0:
        raise ValueError("v1.50 requires period_y_mm = 15.0")
    total_height = (
        float(geometry["main_substrate_thickness_mm"])
        + float(geometry["stack_spacer_thickness_mm"])
        + float(geometry["copper_thickness_mm"])
    )
    if total_height > 3.5:
        raise ValueError("Stacked element exceeds the 3.5 mm height limit")
    sample_count = int(config["doe"]["sample_count"])
    if not 12 <= sample_count <= 20:
        raise ValueError("Periodic DOE sample_count must be in [12, 20]")
    pareto_count = int(config["doe"]["pareto_candidate_count"])
    if not 3 <= pareto_count <= 5:
        raise ValueError("Pareto candidate count must be in [3, 5]")
    if len(config["manufacturing_ranges"]) not in range(10, 13):
        raise ValueError("Exactly 10-12 physical variables may be optimized")
    expected_states = int(
        config["gates"]["required_periodic_scan_state_count"]
    )
    if len(scan_states(config)) != expected_states:
        raise ValueError("Scan-state count does not match the frozen gate")
    if config.get("frequencies_ghz") != FROZEN_FREQUENCIES_GHZ:
        raise ValueError("v1.50 three-frequency grid is immutable")
    if config.get("sweep_ghz") != FROZEN_SWEEP_GHZ:
        raise ValueError("v1.50 frequency sweep is immutable")
    if config.get("scan") != FROZEN_SCAN:
        raise ValueError("v1.50 theta/phi scan grid is immutable")
    if config.get("resources") != FROZEN_RESOURCES:
        raise ValueError("v1.50 resource limits are immutable")
    if config.get("gates") != FROZEN_GATES:
        raise ValueError("v1.50 engineering gates are immutable")
    if float(geometry.get("coax_dielectric_outer_radius_mm", 0.0)) != 0.575:
        raise ValueError("v1.50 coax dielectric radius must be 0.575 mm")
    if float(geometry.get("coax_outer_radius_mm", 0.0)) != 1.05:
        raise ValueError("v1.50 coax outer radius must be 1.05 mm")
    if float(geometry.get("coax_drop_mm", 0.0)) != 1.0:
        raise ValueError("v1.50 coax length must be 1.0 mm")
    if float(geometry.get("ground_thickness_mm", 0.0)) != 0.035:
        raise ValueError("v1.50 ground thickness must be 0.035 mm")
    if int(geometry.get("maximum_adaptive_tetrahedra", 0)) != 120000:
        raise ValueError("v1.50 adaptive tetrahedron limit must be 120000")
    for forbidden in (
        "allow_2x2",
        "allow_4x4",
        "allow_16x16",
        "allow_training_labels",
        "allow_critic_training",
    ):
        if config["scope"].get(forbidden):
            raise ValueError(f"Preregistered scope must lock {forbidden}")


def analytic_vacuum_coax_impedance_ohm(geometry: dict[str, Any]) -> float:
    inner = float(geometry["probe_radius_mm"])
    outer = float(geometry["coax_dielectric_outer_radius_mm"])
    if not 0.0 < inner < outer:
        raise ValueError("Coax radii are not physically nested")
    return 60.0 * math.log(outer / inner)


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def memory_available_gib() -> float:
    if hasattr(ctypes, "windll"):
        status = _MemoryStatus()
        status.length = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.available_physical / (1024.0**3)
    return math.nan


def aedt_processes() -> list[str]:
    result = subprocess.run(
        ["tasklist", "/fo", "csv", "/nh"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "tasklist failed; refusing to assume that no AEDT/HFSS process exists"
        )
    names = (
        "ansysedt.exe",
        "hfss.exe",
        "ansysedtsv.exe",
        "hfsscomengine.exe",
    )
    return [
        line
        for line in result.stdout.splitlines()
        if any(name in line.lower() for name in names)
    ]


@contextmanager
def aedt_execution_lock(lock_path: Path | None = None):
    path = lock_path or (ROOT / ".v150_aedt_execution.lock")
    try:
        descriptor = os.open(
            path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
        )
    except FileExistsError as error:
        raise RuntimeError(
            f"Another v1.50 AEDT orchestration holds {path}"
        ) from error
    try:
        payload = (
            f"pid={os.getpid()}\n"
            f"created_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        ).encode("ascii")
        os.write(descriptor, payload)
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)


def memory_allows_stage(
    config: dict[str, Any],
    free_memory_gib: float,
    aedt_instance_count: int,
) -> bool:
    minimum = float(
        config["resources"]["minimum_free_memory_before_solve_gib"]
    )
    maximum_instances = int(
        config["resources"]["maximum_concurrent_aedt_instances"]
    )
    return (
        math.isfinite(free_memory_gib)
        and free_memory_gib >= minimum
        and aedt_instance_count < maximum_instances
    )


def evaluate_periodic_gate(
    metrics: dict[str, Any], gates: dict[str, Any]
) -> dict[str, Any]:
    finite_fields = (
        "minimum_active_rl_db",
        "broadside_passive_rl_db",
        "minimum_efficiency",
        "maximum_scan_gain_drop_db",
        "maximum_final_delta_s",
    )
    finite_metrics = all(
        isinstance(metrics.get(name), (int, float))
        and math.isfinite(float(metrics[name]))
        for name in finite_fields
    )
    checks = {
        "real_hfss_evidence": (
            metrics.get("evidence_source") == "HFSS_periodic_fullwave"
        ),
        "evidence_complete": metrics.get("evidence_complete") is True,
        "provenance_complete": metrics.get("provenance_complete") is True,
        # Stage A has no independent AEDT-reopen attestor. This check is
        # intentionally impossible to satisfy from caller-provided metrics.
        "physical_attestation": False,
        "stage_a_gate_armed": False,
        "finite_metrics": finite_metrics,
        "scan_state_count": int(metrics.get("scan_state_count", 0))
        >= int(gates["required_periodic_scan_state_count"]),
        "active_rl": float(metrics.get("minimum_active_rl_db", -math.inf))
        >= float(gates["minimum_periodic_active_rl_db"]),
        "broadside_passive_rl": float(
            metrics.get("broadside_passive_rl_db", -math.inf)
        )
        >= float(gates["minimum_broadside_passive_rl_db"]),
        "efficiency": float(metrics.get("minimum_efficiency", -math.inf))
        >= float(gates["minimum_periodic_efficiency"]),
        "scan_gain_drop": float(
            metrics.get("maximum_scan_gain_drop_db", math.inf)
        )
        <= float(gates["maximum_scan_gain_drop_db"]),
        "delta_s": float(metrics.get("maximum_final_delta_s", math.inf))
        <= float(gates["maximum_final_delta_s"]),
        "no_scan_blindness": metrics.get("scan_blindness_detected") is False,
        "critical_warnings": int(
            metrics.get("critical_warning_count", 10**9)
        )
        <= int(gates["maximum_critical_warning_count"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def stage_decision_after_periodic(
    metrics: dict[str, Any], gates: dict[str, Any]
) -> dict[str, Any]:
    gate = evaluate_periodic_gate(metrics, gates)
    return {
        "stage": "periodic_gate_evaluated",
        "periodic_gate_passed": gate["passed"],
        "periodic_gate_checks": gate["checks"],
        "failed_checks": gate["failed_checks"],
        "allow_1x1": gate["passed"],
        "allow_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": (
            "Periodic full-wave gate passed; only the finite 1x1 stage is open."
            if gate["passed"]
            else "Periodic full-wave gate failed or lacks complete evidence; all finite-array and learning stages remain locked."
        ),
    }


def validate_geometry(
    config: dict[str, Any], geometry: dict[str, Any]
) -> None:
    material = config["material"]
    px = float(geometry["period_x_mm"])
    py = float(geometry["period_y_mm"])
    copper = float(geometry["copper_thickness_mm"])
    h_main = float(geometry["main_substrate_thickness_mm"])
    h_stack = float(geometry["stack_spacer_thickness_mm"])
    via_diameter = float(geometry["siw_via_diameter_mm"])
    via_pitch = float(geometry["siw_via_pitch_mm"])
    via_inset = float(geometry["siw_via_inset_mm"])
    top_clearance = float(geometry["siw_top_aperture_clearance_mm"])
    probe = float(geometry["probe_radius_mm"])
    coax_inner = float(geometry["coax_inner_radius_mm"])
    coax_outer = float(geometry["coax_outer_radius_mm"])
    port_overlap = float(config["port_definition"]["contact_overlap_mm"])
    port_plane_offset = float(
        config["port_definition"]["reference_plane_offset_mm"]
    )
    port_axial_height = float(config["port_definition"]["axial_height_mm"])
    coax_drop = float(geometry["coax_drop_mm"])
    feed_x = float(geometry["feed_offset_x_mm"])
    feed_y = float(geometry["feed_offset_y_mm"])
    if px != 15.0 or py != 15.0:
        raise ValueError("Only the frozen 15 mm square lattice is supported")
    if h_main + h_stack + copper > 3.5:
        raise ValueError("Physical stack exceeds 3.5 mm")
    if float(material["relative_permittivity"]) != 2.2:
        raise ValueError("v1.49 material permittivity must remain 2.2")
    if float(material["loss_tangent"]) != 0.0009:
        raise ValueError("v1.49 material loss tangent must remain 0.0009")
    if via_diameter <= 0.0 or via_pitch > 2.0 * via_diameter:
        raise ValueError("SIW via pitch must not exceed twice the diameter")
    if via_inset <= via_diameter / 2.0:
        raise ValueError("SIW via fence does not clear the cell boundary")
    fence_x = px / 2.0 - via_inset
    fence_y = py / 2.0 - via_inset
    clearance = via_diameter / 2.0 + 0.20
    patch_specs = (
        (
            float(geometry["driven_patch_width_mm"]),
            float(geometry["driven_patch_length_mm"]),
        ),
        (
            float(geometry["stacked_patch_width_mm"]),
            float(geometry["stacked_patch_length_mm"]),
        ),
    )
    for width, length in patch_specs:
        if width / 2.0 + clearance >= fence_x:
            raise ValueError("Patch does not clear the x SIW via fence")
        if length / 2.0 + clearance >= fence_y:
            raise ValueError("Patch does not clear the y SIW via fence")
    if top_clearance < 0.20:
        raise ValueError("SIW top aperture needs at least 0.20 mm isolation")
    if (
        patch_specs[0][0] / 2.0 + top_clearance + clearance >= fence_x
        or patch_specs[0][1] / 2.0 + top_clearance + clearance >= fence_y
    ):
        raise ValueError("SIW top aperture does not clear the via fence")
    if not (0.0 < probe < coax_inner < coax_outer):
        raise ValueError("Probe/coax radii are not physically nested")
    if port_overlap != 0.0:
        raise ValueError("Annular coax port must not overlap either conductor")
    if not (
        0.0 < port_plane_offset
        and port_plane_offset + port_axial_height < coax_drop
    ):
        raise ValueError("Port reference plane must lie inside the coax launch")
    if port_axial_height <= 0.0:
        raise ValueError("Lumped-port axial height must be positive")
    if abs(feed_x) + coax_outer >= fence_x:
        raise ValueError("Coax launch does not clear the x SIW fence")
    if abs(feed_y) + coax_outer >= fence_y:
        raise ValueError("Coax launch does not clear the y SIW fence")
    driven_w = float(geometry["driven_patch_width_mm"])
    driven_l = float(geometry["driven_patch_length_mm"])
    if abs(feed_x) + probe >= driven_w / 2.0:
        raise ValueError("Probe does not land on the driven patch in x")
    if abs(feed_y) + probe >= driven_l / 2.0:
        raise ValueError("Probe does not land on the driven patch in y")


def siw_via_centers(geometry: dict[str, Any]) -> list[tuple[float, float]]:
    px = float(geometry["period_x_mm"])
    py = float(geometry["period_y_mm"])
    inset = float(geometry["siw_via_inset_mm"])
    requested_pitch = float(geometry["siw_via_pitch_mm"])
    x_edge = px / 2.0 - inset
    y_edge = py / 2.0 - inset
    nx = max(2, math.ceil((2.0 * x_edge) / requested_pitch))
    ny = max(2, math.ceil((2.0 * y_edge) / requested_pitch))
    points: set[tuple[float, float]] = set()
    for index in range(nx + 1):
        x = -x_edge + 2.0 * x_edge * index / nx
        points.add((round(x, 7), round(-y_edge, 7)))
        points.add((round(x, 7), round(y_edge, 7)))
    for index in range(ny + 1):
        y = -y_edge + 2.0 * y_edge * index / ny
        points.add((round(-x_edge, 7), round(y, 7)))
        points.add((round(x_edge, 7), round(y, 7)))
    return sorted(points)


def geometry_audit(
    config: dict[str, Any], geometry: dict[str, Any]
) -> dict[str, Any]:
    validate_geometry(config, geometry)
    px = float(geometry["period_x_mm"])
    py = float(geometry["period_y_mm"])
    inset = float(geometry["siw_via_inset_mm"])
    radius = float(geometry["siw_via_diameter_mm"]) / 2.0
    clearance_x = px / 2.0 - inset - radius
    clearance_y = py / 2.0 - inset - radius
    feed_clearance = min(
        clearance_x
        - abs(float(geometry["feed_offset_x_mm"]))
        - float(geometry["coax_outer_radius_mm"]),
        clearance_y
        - abs(float(geometry["feed_offset_y_mm"]))
        - float(geometry["coax_outer_radius_mm"]),
    )
    total_height = (
        float(geometry["main_substrate_thickness_mm"])
        + float(geometry["stack_spacer_thickness_mm"])
        + float(geometry["copper_thickness_mm"])
    )
    port_overlap = float(config["port_definition"]["contact_overlap_mm"])
    port_plane_offset = float(
        config["port_definition"]["reference_plane_offset_mm"]
    )
    return {
        "feed_port_count": 1,
        "port_contact_overlap_mm": port_overlap,
        "port_reference_plane_offset_mm": port_plane_offset,
        "port_two_conductor_contact_intended": True,
        "annular_coax_port": False,
        "radial_vertical_lumped_port": True,
        "driven_patch_count": 1,
        "stacked_patch_count": 1,
        "siw_via_count": len(siw_via_centers(geometry)),
        "patches_clear_via_fence": True,
        "feed_clears_siw_vias": feed_clearance > 0.20,
        "total_height_mm": total_height,
        "total_height_gate": total_height <= 3.5,
        "external_decoupling_stage_count": 0,
        "non_neighbor_connection_count": 0,
        "global_0p18mm_mesh_used": False,
        "closed_siw_lower_cavity": True,
    }


def _vp(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\")


def _vbs_helpers() -> str:
    return r'''
Function Mm(value)
    Mm = CStr(Round(CDbl(value), 7)) & "mm"
End Function
Sub CreateBox(editor, objName, x, y, z, dx, dy, dz, material, solveInside)
    editor.CreateBox Array("NAME:BoxParameters", "XPosition:=", Mm(x), "YPosition:=", Mm(y), "ZPosition:=", Mm(z), "XSize:=", Mm(dx), "YSize:=", Mm(dy), "ZSize:=", Mm(dz)), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(190 150 80)", "Transparency:=", 0.35, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub
Sub CreateCylinderZ(editor, objName, x, y, z, radius, height, material, solveInside)
    editor.CreateCylinder Array("NAME:CylinderParameters", "XCenter:=", Mm(x), "YCenter:=", Mm(y), "ZCenter:=", Mm(z), "Radius:=", Mm(radius), "Height:=", Mm(height), "WhichAxis:=", "Z", "NumSides:=", "24"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(220 140 35)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub
Sub CreateSheetZ(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "Z"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(235 150 35)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub
Sub CreateSheetX(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "X"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "NonModel#", "Color:=", "(80 120 255)", "Transparency:=", 0.8, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub
Sub CreateModelSheetY(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "Y"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(235 150 35)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub
Sub SubtractObject(editor, blanks, toolName)
    editor.Subtract Array("NAME:Selections", "Blank Parts:=", blanks, "Tool Parts:=", toolName), Array("NAME:SubtractParameters", "KeepOriginals:=", False)
End Sub
Sub UniteSelection(editor, names)
    editor.Unite Array("NAME:Selections", "Selections:=", names), Array("NAME:UniteParameters", "KeepOriginals:=", False)
End Sub
Sub AssignPort(boundary, portName, sheetName, x1, y1, z1, x2, y2, z2)
    boundary.AssignLumpedPort Array("NAME:" & portName, "Objects:=", Array(sheetName), "DoDeembed:=", False, "RenormalizeAllTerminals:=", True, Array("NAME:Modes", Array("NAME:Mode1", "ModeNum:=", 1, "UseIntLine:=", True, Array("NAME:IntLine", "Coordinate System:=", "Global", "Start:=", Array(Mm(x1), Mm(y1), Mm(z1)), "End:=", Array(Mm(x2), Mm(y2), Mm(z2))), "AlignmentGroup:=", 0, "CharImp:=", "Zpi", "RenormImp:=", "50ohm")), "ShowReporterFilter:=", False, "ReporterFilter:=", Array(True), "Impedance:=", "50ohm")
End Sub
'''


def _element_geometry_text(
    config: dict[str, Any], geometry: dict[str, Any]
) -> tuple[str, list[str]]:
    validate_geometry(config, geometry)
    px = float(geometry["period_x_mm"])
    py = float(geometry["period_y_mm"])
    h_main = float(geometry["main_substrate_thickness_mm"])
    h_stack = float(geometry["stack_spacer_thickness_mm"])
    h_total = h_main + h_stack
    copper = float(geometry["copper_thickness_mm"])
    driven_l = float(geometry["driven_patch_length_mm"])
    driven_w = float(geometry["driven_patch_width_mm"])
    stacked_l = float(geometry["stacked_patch_length_mm"])
    stacked_w = float(geometry["stacked_patch_width_mm"])
    top_clearance = float(geometry["siw_top_aperture_clearance_mm"])
    feed_x = float(geometry["feed_offset_x_mm"])
    feed_y = float(geometry["feed_offset_y_mm"])
    probe = float(geometry["probe_radius_mm"])
    coax_inner = float(geometry["coax_inner_radius_mm"])
    coax_outer = float(geometry["coax_outer_radius_mm"])
    coax_drop = float(geometry["coax_drop_mm"])
    port_z = -coax_drop + float(
        config["port_definition"]["reference_plane_offset_mm"]
    )
    port_height = float(config["port_definition"]["axial_height_mm"])
    port_line_z = port_z + port_height / 2.0
    via_radius = float(geometry["siw_via_diameter_mm"]) / 2.0
    lines = [
        f'CreateBox oEditor, "MainSubstrate", {-px/2:.7f}, {-py/2:.7f}, 0, {px:.7f}, {py:.7f}, {h_main:.7f}, "RO5880_V149", True',
        f'CreateBox oEditor, "StackSpacer", {-px/2:.7f}, {-py/2:.7f}, {h_main:.7f}, {px:.7f}, {py:.7f}, {h_stack:.7f}, "RO5880_V149", True',
        f'CreateSheetZ oEditor, "Ground", {-px/2:.7f}, {-py/2:.7f}, 0, {px:.7f}, {py:.7f}',
        f'CreateCylinderZ oEditor, "GroundFeedCut", {feed_x:.7f}, {feed_y:.7f}, {-copper:.7f}, {coax_inner:.7f}, {2*copper:.7f}, "vacuum", True',
        'SubtractObject oEditor, "Ground", "GroundFeedCut"',
        f'CreateCylinderZ oEditor, "ProbeCut", {feed_x:.7f}, {feed_y:.7f}, -0.01, {probe+0.01:.7f}, {h_main+0.02:.7f}, "vacuum", True',
        'SubtractObject oEditor, "MainSubstrate", "ProbeCut"',
        f'CreateSheetZ oEditor, "SIWCavityTop", {-px/2:.7f}, {-py/2:.7f}, {h_main:.7f}, {px:.7f}, {py:.7f}',
        f'CreateSheetZ oEditor, "CavityPatchAperture", {-driven_w/2-top_clearance:.7f}, {-driven_l/2-top_clearance:.7f}, {h_main:.7f}, {driven_w+2*top_clearance:.7f}, {driven_l+2*top_clearance:.7f}',
        'SubtractObject oEditor, "SIWCavityTop", "CavityPatchAperture"',
        f'CreateSheetZ oEditor, "DrivenPatch", {-driven_w/2:.7f}, {-driven_l/2:.7f}, {h_main:.7f}, {driven_w:.7f}, {driven_l:.7f}',
        f'CreateSheetZ oEditor, "StackedPatch", {-stacked_w/2:.7f}, {-stacked_l/2:.7f}, {h_total:.7f}, {stacked_w:.7f}, {stacked_l:.7f}',
        f'CreateCylinderZ oEditor, "FeedProbe", {feed_x:.7f}, {feed_y:.7f}, {-coax_drop:.7f}, {probe:.7f}, {coax_drop+h_main:.7f}, "copper", False',
        f'CreateCylinderZ oEditor, "CoaxOuter", {feed_x:.7f}, {feed_y:.7f}, {-coax_drop:.7f}, {coax_outer:.7f}, {coax_drop:.7f}, "copper", False',
        f'CreateCylinderZ oEditor, "CoaxOuterCut", {feed_x:.7f}, {feed_y:.7f}, {-coax_drop-0.01:.7f}, {coax_inner:.7f}, {coax_drop+0.02:.7f}, "vacuum", True',
        'SubtractObject oEditor, "CoaxOuter", "CoaxOuterCut"',
        f'CreateBox oEditor, "CoaxDielectric", {feed_x-coax_inner:.7f}, {feed_y-coax_inner:.7f}, {-coax_drop:.7f}, {2*coax_inner:.7f}, {2*coax_inner:.7f}, {coax_drop:.7f}, "vacuum", True',
        f'CreateCylinderZ oEditor, "CoaxDielectricTrim", {feed_x:.7f}, {feed_y:.7f}, {-coax_drop-0.01:.7f}, {coax_inner:.7f}, {coax_drop+0.02:.7f}, "vacuum", True',
        'oEditor.Intersect Array("NAME:Selections", "Selections:=", "CoaxDielectric,CoaxDielectricTrim"), Array("NAME:IntersectParameters", "KeepOriginals:=", False)',
        f'CreateCylinderZ oEditor, "CoaxProbeCut", {feed_x:.7f}, {feed_y:.7f}, {-coax_drop-0.01:.7f}, {probe:.7f}, {coax_drop+0.02:.7f}, "vacuum", True',
        'SubtractObject oEditor, "CoaxDielectric", "CoaxProbeCut"',
        f'CreateModelSheetY oEditor, "PortSheet", {feed_x+probe:.7f}, {feed_y:.7f}, {port_z:.7f}, {port_height:.7f}, {coax_inner-probe:.7f}',
        f'AssignPort oBoundary, "FeedPort", "PortSheet", {feed_x+probe:.7f}, {feed_y:.7f}, {port_line_z:.7f}, {feed_x+coax_inner:.7f}, {feed_y:.7f}, {port_line_z:.7f}',
    ]
    via_names = []
    for index, (x, y) in enumerate(siw_via_centers(geometry)):
        name = f"SIWVia_{index:03d}"
        cut = f"SIWViaCut_{index:03d}"
        via_names.append(name)
        lines.extend(
            [
                f'CreateCylinderZ oEditor, "{cut}", {x:.7f}, {y:.7f}, -0.01, {via_radius+0.01:.7f}, {h_main+0.02:.7f}, "vacuum", True',
                f'SubtractObject oEditor, "MainSubstrate", "{cut}"',
                f'CreateCylinderZ oEditor, "{name}", {x:.7f}, {y:.7f}, 0, {via_radius:.7f}, {h_main:.7f}, "copper", False',
            ]
        )
    finite_objects = (
        '"Ground", "SIWCavityTop", "DrivenPatch", "StackedPatch"'
    )
    lines.append(
        'oBoundary.AssignFiniteCond Array("NAME:CopperSheetFiniteConductivity", '
        f'"Objects:=", Array({finite_objects}), "UseMaterial:=", True, '
        '"Material:=", "copper", "UseThickness:=", True, '
        f'"Thickness:=", "{copper:.7f}mm", "Roughness:=", "0um", '
        '"InfGroundPlane:=", False, "IsTwoSided:=", True, '
        '"IsShellElement:=", False)'
    )
    return "\n".join(lines), via_names


def _project_header(config: dict[str, Any], design_name: str) -> str:
    material = config["material"]
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oMesh
Dim fso, auditFile, validationFile, validationPassed, objectNames, boundaryNames, excitationNames, i
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO5880_V149", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "{float(material['relative_permittivity']):g}", "dielectric_loss_tangent:=", "{float(material['loss_tangent']):g}")
oProject.InsertDesign "HFSS", "{design_name}", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("{design_name}")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
Set oMesh = oDesign.GetModule("MeshSetup")
'''


def _mesh_and_setup_text(
    config: dict[str, Any],
    geometry: dict[str, Any],
    via_names: list[str],
    frequency_ghz: float,
    solver_type: str,
) -> str:
    solver = (
        "Domain Decomposition" if solver_type == "ddm" else "Direct Solver"
    )
    via_array = ", ".join(f'"{name}"' for name in via_names)
    return f'''
oMesh.AssignLengthOp Array("NAME:Mesh_ProbeLaunch", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array("FeedProbe", "CoaxOuter"), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{float(geometry['local_mesh_probe_mm']):.7f}mm", "UseAdvSizing:=", False)
oMesh.AssignLengthOp Array("NAME:Mesh_PortSheet", "RefineInside:=", True, "Enabled:=", True, "Objects:=", Array("PortSheet"), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{float(config['port_definition']['local_mesh_mm']):.7f}mm", "UseAdvSizing:=", False)
oMesh.AssignLengthOp Array("NAME:Mesh_PatchEdges", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array("DrivenPatch", "StackedPatch"), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{float(geometry['local_mesh_patch_edge_mm']):.7f}mm", "UseAdvSizing:=", False)
oMesh.AssignLengthOp Array("NAME:Mesh_SIWVias", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array({via_array}), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "{float(geometry['local_mesh_via_mm']):.7f}mm", "UseAdvSizing:=", False)
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "{frequency_ghz:g}GHz", "MaxDeltaS:=", 0.05, "MaximumPasses:=", {int(geometry['maximum_passes'])}, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", 2, "PercentRefinement:=", {float(geometry['adaptive_refinement_percent']):.7f}, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "SetLambdaTarget:=", False, "UseMaxTetIncrease:=", False, "PortAccuracy:=", 2, "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False, "DrivenSolverType:=", "{solver}")
'''


def _frequency_sweep_text(config: dict[str, Any]) -> str:
    sweep = config["sweep_ghz"]
    return f'''
oAnalysis.InsertFrequencySweep "Setup_10GHz", Array("NAME:Sweep_9p8_10p2", "IsEnabled:=", True, "RangeType:=", "LinearStep", "RangeStart:=", "{float(sweep['start']):g}GHz", "RangeEnd:=", "{float(sweep['stop']):g}GHz", "RangeStep:=", "{float(sweep['step']):g}GHz", "Type:=", "Interpolating", "SaveFields:=", False, "SaveRadFields:=", False, "InterpTolerance:=", 0.5, "InterpMaxSolns:=", 250, "InterpMinSolns:=", 0, "InterpMinSubranges:=", 1, "ExtrapToDC:=", False, "InterpUseS:=", True, "InterpUsePortImped:=", True, "InterpUsePropConst:=", True, "UseDerivativeConvergence:=", False, "InterpDerivTolerance:=", 0.2, "UseFullBasis:=", True, "EnforcePassivity:=", True, "PassivityErrorTolerance:=", 0.0001)
'''


def periodic_builder_text(
    project: Path,
    config: dict[str, Any],
    geometry: dict[str, Any],
    state: dict[str, float],
) -> str:
    body, via_names = _element_geometry_text(config, geometry)
    px = float(geometry["period_x_mm"])
    py = float(geometry["period_y_mm"])
    h_main = float(geometry["main_substrate_thickness_mm"])
    h_stack = float(geometry["stack_spacer_thickness_mm"])
    h_total = h_main + h_stack
    air_height = float(geometry["air_above_mm"])
    top = (
        h_total
        + air_height
    )
    bottom = 0.0
    sample_z = h_total + air_height / 2.0
    theta = float(state["theta_deg"])
    phi = float(state["phi_deg"])
    frequency = float(state["frequency_ghz"])
    inventory = project.parent / "model_inventory.txt"
    validation = project.parent / "design_validation.txt"
    return (
        _project_header(config, "V149_PeriodicUnit")
        + body
        + f'''
CreateBox oEditor, "AirCell", {-px/2:.7f}, {-py/2:.7f}, {h_total:.7f}, {px:.7f}, {py:.7f}, {air_height:.7f}, "vacuum", True
Dim primaryXFace, secondaryXFace, primaryYFace, secondaryYFace, primaryXMainFace, secondaryXMainFace, primaryYMainFace, secondaryYMainFace, primaryXStackFace, secondaryXStackFace, primaryYStackFace, secondaryYStackFace, floquetFace
primaryXFace = oEditor.GetFaceByPosition(Array("NAME:FaceParameters", "BodyName:=", "AirCell", "XPosition:=", Mm({-px/2:.7f}), "YPosition:=", Mm(0), "ZPosition:=", Mm({sample_z:.7f})))
secondaryXFace = oEditor.GetFaceByPosition(Array("NAME:FaceParameters", "BodyName:=", "AirCell", "XPosition:=", Mm({px/2:.7f}), "YPosition:=", Mm(0), "ZPosition:=", Mm({sample_z:.7f})))
primaryYFace = oEditor.GetFaceByPosition(Array("NAME:FaceParameters", "BodyName:=", "AirCell", "XPosition:=", Mm(0), "YPosition:=", Mm({-py/2:.7f}), "ZPosition:=", Mm({sample_z:.7f})))
secondaryYFace = oEditor.GetFaceByPosition(Array("NAME:FaceParameters", "BodyName:=", "AirCell", "XPosition:=", Mm(0), "YPosition:=", Mm({py/2:.7f}), "ZPosition:=", Mm({sample_z:.7f})))
primaryXMainFace = oEditor.GetFaceByPosition(Array("NAME:FaceParameters", "BodyName:=", "MainSubstrate", "XPosition:=", Mm({-px/2:.7f}), "YPosition:=", Mm(0), "ZPosition:=", Mm({h_main/2:.7f})))
secondaryXMainFace = oEditor.GetFaceByPosition(Array("NAME:FaceParameters", "BodyName:=", "MainSubstrate", "XPosition:=", Mm({px/2:.7f}), "YPosition:=", Mm(0), "ZPosition:=", Mm({h_main/2:.7f})))
primaryYMainFace = oEditor.GetFaceByPosition(Array("NAME:FaceParameters", "BodyName:=", "MainSubstrate", "XPosition:=", Mm(0), "YPosition:=", Mm({-py/2:.7f}), "ZPosition:=", Mm({h_main/2:.7f})))
secondaryYMainFace = oEditor.GetFaceByPosition(Array("NAME:FaceParameters", "BodyName:=", "MainSubstrate", "XPosition:=", Mm(0), "YPosition:=", Mm({py/2:.7f}), "ZPosition:=", Mm({h_main/2:.7f})))
primaryXStackFace = oEditor.GetFaceByPosition(Array("NAME:FaceParameters", "BodyName:=", "StackSpacer", "XPosition:=", Mm({-px/2:.7f}), "YPosition:=", Mm(0), "ZPosition:=", Mm({h_main+h_stack/2:.7f})))
secondaryXStackFace = oEditor.GetFaceByPosition(Array("NAME:FaceParameters", "BodyName:=", "StackSpacer", "XPosition:=", Mm({px/2:.7f}), "YPosition:=", Mm(0), "ZPosition:=", Mm({h_main+h_stack/2:.7f})))
primaryYStackFace = oEditor.GetFaceByPosition(Array("NAME:FaceParameters", "BodyName:=", "StackSpacer", "XPosition:=", Mm(0), "YPosition:=", Mm({-py/2:.7f}), "ZPosition:=", Mm({h_main+h_stack/2:.7f})))
secondaryYStackFace = oEditor.GetFaceByPosition(Array("NAME:FaceParameters", "BodyName:=", "StackSpacer", "XPosition:=", Mm(0), "YPosition:=", Mm({py/2:.7f}), "ZPosition:=", Mm({h_main+h_stack/2:.7f})))
floquetFace = oEditor.GetFaceByPosition(Array("NAME:FaceParameters", "BodyName:=", "AirCell", "XPosition:=", Mm(0), "YPosition:=", Mm(0), "ZPosition:=", Mm({top:.7f})))
oBoundary.AssignPrimary Array("NAME:PrimaryX_Main", Array("NAME:CoordSysVector", "Origin:=", Array(Mm({-px/2:.7f}), Mm({-py/2:.7f}), Mm({bottom:.7f})), "UPos:=", Array(Mm({-px/2:.7f}), Mm({py/2:.7f}), Mm({bottom:.7f}))), "ReverseV:=", False, "Faces:=", Array(CLng(primaryXMainFace)))
oBoundary.AssignSecondary Array("NAME:SecondaryX_Main", Array("NAME:CoordSysVector", "Origin:=", Array(Mm({px/2:.7f}), Mm({-py/2:.7f}), Mm({bottom:.7f})), "UPos:=", Array(Mm({px/2:.7f}), Mm({py/2:.7f}), Mm({bottom:.7f}))), "ReverseV:=", True, "Primary:=", "PrimaryX_Main", "UseScanAngles:=", True, "Phi:=", "{phi:g}deg", "Theta:=", "{theta:g}deg", "Faces:=", Array(CLng(secondaryXMainFace)))
oBoundary.AssignPrimary Array("NAME:PrimaryX_Stack", Array("NAME:CoordSysVector", "Origin:=", Array(Mm({-px/2:.7f}), Mm({-py/2:.7f}), Mm({h_main:.7f})), "UPos:=", Array(Mm({-px/2:.7f}), Mm({py/2:.7f}), Mm({h_main:.7f}))), "ReverseV:=", False, "Faces:=", Array(CLng(primaryXStackFace)))
oBoundary.AssignSecondary Array("NAME:SecondaryX_Stack", Array("NAME:CoordSysVector", "Origin:=", Array(Mm({px/2:.7f}), Mm({-py/2:.7f}), Mm({h_main:.7f})), "UPos:=", Array(Mm({px/2:.7f}), Mm({py/2:.7f}), Mm({h_main:.7f}))), "ReverseV:=", True, "Primary:=", "PrimaryX_Stack", "UseScanAngles:=", True, "Phi:=", "{phi:g}deg", "Theta:=", "{theta:g}deg", "Faces:=", Array(CLng(secondaryXStackFace)))
oBoundary.AssignPrimary Array("NAME:PrimaryX", Array("NAME:CoordSysVector", "Origin:=", Array(Mm({-px/2:.7f}), Mm({-py/2:.7f}), Mm({h_total:.7f})), "UPos:=", Array(Mm({-px/2:.7f}), Mm({py/2:.7f}), Mm({h_total:.7f}))), "ReverseV:=", False, "Faces:=", Array(CLng(primaryXFace)))
oBoundary.AssignSecondary Array("NAME:SecondaryX", Array("NAME:CoordSysVector", "Origin:=", Array(Mm({px/2:.7f}), Mm({-py/2:.7f}), Mm({h_total:.7f})), "UPos:=", Array(Mm({px/2:.7f}), Mm({py/2:.7f}), Mm({h_total:.7f}))), "ReverseV:=", True, "Primary:=", "PrimaryX", "UseScanAngles:=", True, "Phi:=", "{phi:g}deg", "Theta:=", "{theta:g}deg", "Faces:=", Array(CLng(secondaryXFace)))
oBoundary.AssignPrimary Array("NAME:PrimaryY_Main", Array("NAME:CoordSysVector", "Origin:=", Array(Mm({-px/2:.7f}), Mm({-py/2:.7f}), Mm({bottom:.7f})), "UPos:=", Array(Mm({px/2:.7f}), Mm({-py/2:.7f}), Mm({bottom:.7f}))), "ReverseV:=", False, "Faces:=", Array(CLng(primaryYMainFace)))
oBoundary.AssignSecondary Array("NAME:SecondaryY_Main", Array("NAME:CoordSysVector", "Origin:=", Array(Mm({-px/2:.7f}), Mm({py/2:.7f}), Mm({bottom:.7f})), "UPos:=", Array(Mm({px/2:.7f}), Mm({py/2:.7f}), Mm({bottom:.7f}))), "ReverseV:=", True, "Primary:=", "PrimaryY_Main", "UseScanAngles:=", True, "Phi:=", "{phi:g}deg", "Theta:=", "{theta:g}deg", "Faces:=", Array(CLng(secondaryYMainFace)))
oBoundary.AssignPrimary Array("NAME:PrimaryY_Stack", Array("NAME:CoordSysVector", "Origin:=", Array(Mm({-px/2:.7f}), Mm({-py/2:.7f}), Mm({h_main:.7f})), "UPos:=", Array(Mm({px/2:.7f}), Mm({-py/2:.7f}), Mm({h_main:.7f}))), "ReverseV:=", False, "Faces:=", Array(CLng(primaryYStackFace)))
oBoundary.AssignSecondary Array("NAME:SecondaryY_Stack", Array("NAME:CoordSysVector", "Origin:=", Array(Mm({-px/2:.7f}), Mm({py/2:.7f}), Mm({h_main:.7f})), "UPos:=", Array(Mm({px/2:.7f}), Mm({py/2:.7f}), Mm({h_main:.7f}))), "ReverseV:=", True, "Primary:=", "PrimaryY_Stack", "UseScanAngles:=", True, "Phi:=", "{phi:g}deg", "Theta:=", "{theta:g}deg", "Faces:=", Array(CLng(secondaryYStackFace)))
oBoundary.AssignPrimary Array("NAME:PrimaryY", Array("NAME:CoordSysVector", "Origin:=", Array(Mm({-px/2:.7f}), Mm({-py/2:.7f}), Mm({h_total:.7f})), "UPos:=", Array(Mm({px/2:.7f}), Mm({-py/2:.7f}), Mm({h_total:.7f}))), "ReverseV:=", False, "Faces:=", Array(CLng(primaryYFace)))
oBoundary.AssignSecondary Array("NAME:SecondaryY", Array("NAME:CoordSysVector", "Origin:=", Array(Mm({-px/2:.7f}), Mm({py/2:.7f}), Mm({h_total:.7f})), "UPos:=", Array(Mm({px/2:.7f}), Mm({py/2:.7f}), Mm({h_total:.7f}))), "ReverseV:=", True, "Primary:=", "PrimaryY", "UseScanAngles:=", True, "Phi:=", "{phi:g}deg", "Theta:=", "{theta:g}deg", "Faces:=", Array(CLng(secondaryYFace)))
oBoundary.AssignFloquetPort Array("NAME:FloquetTop", "Faces:=", Array(CLng(floquetFace)), "NumModes:=", 2, "RenormalizeAllTerminals:=", True, "DoDeembed:=", False, Array("NAME:Modes", Array("NAME:Mode1", "ModeNum:=", 1, "UseIntLine:=", False), Array("NAME:Mode2", "ModeNum:=", 2, "UseIntLine:=", False)), "ShowReporterFilter:=", False, "UseScanAngles:=", True, "Phi:=", "{phi:g}deg", "Theta:=", "{theta:g}deg", Array("NAME:LatticeAVector", "Start:=", Array(Mm({-px/2:.7f}), Mm({-py/2:.7f}), Mm({top:.7f})), "End:=", Array(Mm({px/2:.7f}), Mm({-py/2:.7f}), Mm({top:.7f}))), Array("NAME:LatticeBVector", "Start:=", Array(Mm({-px/2:.7f}), Mm({-py/2:.7f}), Mm({top:.7f})), "End:=", Array(Mm({-px/2:.7f}), Mm({py/2:.7f}), Mm({top:.7f}))), Array("NAME:ModesCalculator", "Frequency:=", "{frequency:g}GHz", "FrequencyChanged:=", False, "PhiStart:=", "{phi:g}deg", "PhiStop:=", "{phi:g}deg", "PhiStep:=", "0deg", "ThetaStart:=", "{theta:g}deg", "ThetaStop:=", "{theta:g}deg", "ThetaStep:=", "0deg"), Array("NAME:ModesList", Array("NAME:Mode", "ModeNumber:=", 1, "IndexM:=", 0, "IndexN:=", 0, "KC2:=", 0, "PropagationState:=", "Propagating", "Attenuation:=", 0, "PolarizationState:=", "TE", "AffectsRefinement:=", False), Array("NAME:Mode", "ModeNumber:=", 2, "IndexM:=", 0, "IndexN:=", 0, "KC2:=", 0, "PropagationState:=", "Propagating", "Attenuation:=", 0, "PolarizationState:=", "TM", "AffectsRefinement:=", False)))
'''
        + _mesh_and_setup_text(
            config, geometry, via_names, frequency, "direct"
        )
        + _frequency_sweep_text(config)
        + f'''
validationPassed = oDesign.ValidateDesign()
Set fso = CreateObject("Scripting.FileSystemObject")
Set validationFile = fso.CreateTextFile("{_vp(validation)}", True)
validationFile.WriteLine "VALIDATION|" & CStr(validationPassed)
validationFile.Close
oProject.SaveAs "{_vp(project)}", True
objectNames = oEditor.GetMatchedObjectName("*")
boundaryNames = oBoundary.GetBoundaries()
excitationNames = oBoundary.GetExcitations()
Set auditFile = fso.CreateTextFile("{_vp(inventory)}", True)
For i = LBound(objectNames) To UBound(objectNames)
    auditFile.WriteLine "OBJECT|" & CStr(objectNames(i))
Next
For i = LBound(boundaryNames) To UBound(boundaryNames)
    auditFile.WriteLine "BOUNDARY|" & CStr(boundaryNames(i))
Next
For i = LBound(excitationNames) To UBound(excitationNames) Step 2
    auditFile.WriteLine "EXCITATION|" & CStr(excitationNames(i)) & "|" & CStr(excitationNames(i + 1))
Next
auditFile.Close
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''
        + _vbs_helpers()
    )


def finite_one_by_one_builder_text(
    project: Path,
    config: dict[str, Any],
    geometry: dict[str, Any],
    solver_type: str,
) -> str:
    body, via_names = _element_geometry_text(config, geometry)
    frequency = float(config["frequencies_ghz"][1])
    padding = max(
        8.0,
        float(geometry["air_above_mm"]),
        float(geometry["air_below_mm"]),
    )
    return (
        _project_header(config, "V149_Finite1x1")
        + body
        + f'''
oEditor.CreateRegion Array("NAME:RegionParameters", "+XPaddingType:=", "Absolute Offset", "+XPadding:=", "{padding:g}mm", "-XPaddingType:=", "Absolute Offset", "-XPadding:=", "{padding:g}mm", "+YPaddingType:=", "Absolute Offset", "+YPadding:=", "{padding:g}mm", "-YPaddingType:=", "Absolute Offset", "-YPadding:=", "{padding:g}mm", "+ZPaddingType:=", "Absolute Offset", "+ZPadding:=", "{padding:g}mm", "-ZPaddingType:=", "Absolute Offset", "-ZPadding:=", "{padding:g}mm"), Array("NAME:Attributes", "Name:=", "AirRegion", "Flags:=", "Wireframe#", "Color:=", "(128 128 255)", "Transparency:=", 0.9, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """air""", "SolveInside:=", True)
oBoundary.AssignRadiation Array("NAME:Radiation_AirRegion", "Objects:=", Array("AirRegion"))
'''
        + _mesh_and_setup_text(
            config, geometry, via_names, frequency, solver_type
        )
        + _frequency_sweep_text(config)
        + f'''
oProject.SaveAs "{_vp(project)}", True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''
        + _vbs_helpers()
    )


def solve_preflight(
    config: dict[str, Any],
    free_memory_gib: float,
    aedt_instance_count: int,
) -> dict[str, Any]:
    minimum = float(
        config["resources"]["minimum_free_memory_before_solve_gib"]
    )
    maximum = int(
        config["resources"]["maximum_concurrent_aedt_instances"]
    )
    if not math.isfinite(free_memory_gib) or free_memory_gib < minimum:
        return {
            "allowed": False,
            "reason": (
                f"Available memory {free_memory_gib:.2f} GiB is below the "
                f"{minimum:.2f} GiB full-wave solve threshold."
            ),
        }
    if aedt_instance_count >= maximum:
        return {
            "allowed": False,
            "reason": (
                f"{aedt_instance_count} AEDT/HFSS instance(s) already exist; "
                "serial execution requires zero."
            ),
        }
    return {
        "allowed": True,
        "reason": "Full-wave memory and serial-process gates pass.",
    }


def periodic_solver_text(
    project: Path,
    touchstone: Path,
    source_names: Path,
    config: dict[str, Any],
) -> str:
    return f'''Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oAnalysis, oSolutions
Dim vars, variation, sources, fso, sourceFile, i
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "{_vp(project)}"
Set oProject = oDesktop.SetActiveProject("{project.stem}")
Set oDesign = oProject.SetActiveDesign("V149_PeriodicUnit")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
On Error Resume Next
{_frequency_sweep_text(config).strip()}
Err.Clear
On Error GoTo 0
oDesign.Analyze "Setup_10GHz"
oProject.Save
Set oSolutions = oDesign.GetModule("Solutions")
vars = oSolutions.ListVariations("Setup_10GHz:LastAdaptive")
variation = CStr(vars(LBound(vars)))
oSolutions.ExportNetworkData variation, Array("Setup_10GHz:Sweep_9p8_10p2"), 3, "{_vp(touchstone)}", Array("All"), True, 50, "S", -1, 0, 15, True, True, False
sources = oSolutions.GetAllSources()
Set fso = CreateObject("Scripting.FileSystemObject")
Set sourceFile = fso.CreateTextFile("{_vp(source_names)}", True)
For i = LBound(sources) To UBound(sources)
    sourceFile.WriteLine CStr(sources(i))
Next
sourceFile.Close
oProject.Save
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
'''


def allocate_output_root(config: dict[str, Any]) -> Path:
    prefix = resolve(config["output_prefix"])
    for index in range(1, 100):
        candidate = Path(f"{prefix}{index:02d}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError("No free v1.49 run index remains")


def preregister(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    executable = resolve(config["ansys_executable"])
    if not executable.exists():
        raise FileNotFoundError(executable)
    baseline = str(config["baseline_commit"])
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", baseline, "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("Frozen v1.49 baseline is not an ancestor of HEAD")
    inputs = {name: resolve(path) for name, path in config["inputs"].items()}
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen v1.49 input(s): {missing}")
    root = allocate_output_root(config)
    (root / "logs").mkdir()
    (root / "manifests").mkdir()
    frozen = root / "frozen_inputs"
    frozen.mkdir()
    copied: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    for name, source in inputs.items():
        destination = frozen / source.name
        shutil.copy2(source, destination)
        input_hashes[name] = sha256(source)
        copied[name] = str(destination.resolve())
        if sha256(destination) != input_hashes[name]:
            raise RuntimeError(f"Frozen input hash mismatch: {name}")
    preregistration = dict(config)
    preregistration["allocated_output_root"] = str(root.resolve())
    preregistration["preregistered_at"] = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    write_json(root / "preregistration.json", preregistration)
    preregistration_hash = sha256(root / "preregistration.json")
    audit = {
        "protocol": config["protocol"],
        "baseline_commit": baseline,
        "head_commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "working_tree_status": git("status", "--short", check=False),
        "ansys_executable": str(executable.resolve()),
        "ansys_executable_sha256": sha256(executable),
        "input_sha256": input_hashes,
        "frozen_inputs": copied,
        "free_memory_gib": memory_available_gib(),
        "aedt_processes": aedt_processes(),
        "disk_free_gib": shutil.disk_usage(root).free / (1024.0**3),
        "preregistration_sha256": preregistration_hash,
    }
    write_json(root / "baseline_audit.json", audit)
    decision = {
        "stage": "A_periodic_build_only",
        "allow_periodic_build_smoke": True,
        "allow_periodic_nominal_solve": False,
        "allow_periodic_doe_batch": False,
        "allow_1x1": False,
        "allow_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": (
            "v1.49 is preregistered. Only native periodic CAD generation and "
            "a build-only smoke are authorized."
        ),
    }
    write_json(root / "stage_decision.json", decision)
    return {
        "output_root": str(root.resolve()),
        "audit": audit,
        "decision": decision,
    }


def control_script_provenance() -> dict[str, Any]:
    script = Path(__file__).resolve()
    relative = script.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    head_sha256 = (
        hashlib.sha256(result.stdout).hexdigest()
        if result.returncode == 0
        else None
    )
    diff = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", relative],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    current_sha256 = sha256(script)
    return {
        "path": str(script),
        "sha256": current_sha256,
        "head_sha256": head_sha256,
        "git_diff_returncode": diff.returncode,
        "tracked_at_head": result.returncode == 0 and diff.returncode == 0,
    }


def _path_within(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"{label} is outside its authorized root") from error
    return resolved


def seal_nominal_authorization(
    run_root: Path,
    manifest_path: Path,
    authorization_kind: str,
    source_evidence: dict[str, Any],
) -> dict[str, Any]:
    run_root = run_root.resolve()
    provenance = control_script_provenance()
    if provenance.get("tracked_at_head") is not True:
        raise RuntimeError(
            "Continuation control script must be committed at HEAD before "
            "an executable run is allocated"
        )
    decision_path = run_root / "stage_decision.json"
    baseline_path = run_root / "baseline_audit.json"
    preregistration_path = run_root / "preregistration.json"
    continuation_path = run_root / "continuation_audit.json"
    for required in (
        manifest_path,
        decision_path,
        baseline_path,
        preregistration_path,
    ):
        if not required.is_file():
            raise RuntimeError(
                f"Cannot seal nominal authorization without {required.name}"
            )
    snapshot_folder = run_root / "control_snapshot"
    snapshot_folder.mkdir(exist_ok=False)
    snapshot = snapshot_folder / Path(provenance["path"]).name
    shutil.copy2(Path(provenance["path"]), snapshot)
    if sha256(snapshot) != provenance["sha256"]:
        raise RuntimeError("Control-script snapshot hash mismatch")
    authorization = {
        "schema": "v149_nominal_solve_authorization_v1",
        "authorization_kind": authorization_kind,
        "issued_for_run": str(run_root),
        "issued_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "head_commit": git("rev-parse", "HEAD"),
        "control_snapshot_path": str(snapshot.resolve()),
        "control_snapshot_sha256": sha256(snapshot),
        "control_script_head_sha256": provenance["head_sha256"],
        "case_manifest_sha256": sha256(manifest_path),
        "stage_decision_sha256": sha256(decision_path),
        "baseline_audit_sha256": sha256(baseline_path),
        "preregistration_sha256": sha256(preregistration_path),
        "continuation_audit_sha256": (
            sha256(continuation_path) if continuation_path.is_file() else None
        ),
        "source_evidence": source_evidence,
        "single_use": True,
    }
    authorization_path = run_root / "solve_authorization.json"
    if authorization_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite solve authorization: {authorization_path}"
        )
    write_json(authorization_path, authorization)
    authorization["authorization_path"] = str(authorization_path.resolve())
    authorization["authorization_sha256"] = sha256(authorization_path)
    return authorization


def verify_finalized_run_manifest(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    summary = run_root / "stage_summary.json"
    manifest = run_root / "sha256_manifest.csv"
    if not summary.exists() or not manifest.exists():
        raise RuntimeError(
            "Continuation source must be a finalized run with stage summary "
            "and SHA-256 manifest"
        )
    checked: list[str] = []
    seen: set[str] = set()
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            relative = row.get("relative_path", "")
            if not relative or relative in seen:
                raise RuntimeError("Invalid finalized source SHA-256 manifest")
            seen.add(relative)
            path = (run_root / relative).resolve()
            try:
                path.relative_to(run_root)
            except ValueError as error:
                raise RuntimeError(
                    "Finalized source SHA-256 manifest escapes the run root"
                ) from error
            if not path.is_file():
                raise RuntimeError(
                    f"Finalized source SHA-256 manifest file is missing: {relative}"
                )
            if int(row.get("bytes", -1)) != path.stat().st_size or row.get(
                "sha256"
            ) != sha256(path):
                raise RuntimeError(
                    f"Finalized source SHA-256 manifest mismatch: {relative}"
                )
            checked.append(relative)
    if not checked:
        raise RuntimeError("Finalized source SHA-256 manifest is empty")
    return {
        "manifest_path": str(manifest),
        "manifest_sha256": sha256(manifest),
        "verified_file_count": len(checked),
    }


def create_nominal_continuation(source_run_root: Path) -> dict[str, Any]:
    source_run_root = source_run_root.resolve()
    source_manifest_audit = verify_finalized_run_manifest(source_run_root)
    config = load_run_config(source_run_root)
    build_folder = source_run_root / "periodic" / "build_smoke"
    build_gate_path = build_folder / "build_gate.json"
    build_manifest_path = build_folder / "case_manifest.json"
    if not build_gate_path.exists() or not build_manifest_path.exists():
        raise RuntimeError("Finalized source is missing build-gate evidence")
    build_gate = json.loads(build_gate_path.read_text(encoding="utf-8"))
    if build_gate.get("build_smoke_passed") is not True:
        raise RuntimeError("Continuation source build smoke did not pass")
    if build_gate.get("model_inventory_verified") is not True:
        raise RuntimeError("Continuation source model inventory is unverified")
    build_manifest = json.loads(
        build_manifest_path.read_text(encoding="utf-8")
    )
    source_project = Path(build_manifest["project_path"]).resolve()
    _path_within(
        source_project, source_run_root, "Continuation source project"
    )
    if not source_project.is_file():
        raise FileNotFoundError(source_project)
    source_project_sha256 = sha256(source_project)
    if build_gate.get("project_sha256") != source_project_sha256:
        raise RuntimeError("Continuation source build project hash mismatch")
    inventory_path = _path_within(
        Path(build_manifest.get("model_inventory_path", "")),
        source_run_root,
        "Continuation model inventory",
    )
    if not inventory_path.is_file():
        raise RuntimeError("Continuation source model inventory is missing")
    inventory = validate_model_inventory(
        inventory_path.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines(),
        config,
        config["nominal_geometry"],
    )
    if inventory.get("verified") is not True:
        raise RuntimeError(
            "Continuation source model inventory failed independent validation"
        )
    inventory_sha256 = sha256(inventory_path)
    if control_script_provenance().get("tracked_at_head") is not True:
        raise RuntimeError(
            "Continuation control script must be committed at HEAD before "
            "allocating a new run"
        )

    root = allocate_output_root(config)
    (root / "logs").mkdir()
    (root / "manifests").mkdir()
    frozen = root / "frozen_inputs"
    frozen.mkdir()
    copied: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    source_audit = json.loads(
        (source_run_root / "baseline_audit.json").read_text(encoding="utf-8")
    )
    expected_input_hashes = source_audit.get("input_sha256", {})
    for name, original in config["inputs"].items():
        source = source_run_root / "frozen_inputs" / Path(original).name
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hash = sha256(source)
        if expected_input_hashes.get(name) != source_hash:
            raise RuntimeError(f"Frozen continuation input mismatch: {name}")
        destination = frozen / source.name
        shutil.copy2(source, destination)
        if sha256(destination) != source_hash:
            raise RuntimeError(f"Copied continuation input mismatch: {name}")
        input_hashes[name] = source_hash
        copied[name] = str(destination.resolve())

    preregistration = dict(config)
    preregistration["allocated_output_root"] = str(root.resolve())
    preregistration["preregistered_at"] = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    preregistration["continuation_source_run"] = str(source_run_root)
    preregistration["continuation_scope"] = "one_nominal_periodic_sweep"
    preregistration_path = root / "preregistration.json"
    write_json(preregistration_path, preregistration)
    baseline_audit = {
        "protocol": config["protocol"],
        "continuation": True,
        "source_run": str(source_run_root),
        "source_manifest_sha256": source_manifest_audit["manifest_sha256"],
        "source_build_gate_sha256": sha256(build_gate_path),
        "source_build_project_sha256": source_project_sha256,
        "baseline_commit": config["baseline_commit"],
        "head_commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "working_tree_status": git("status", "--short", check=False),
        "ansys_executable": str(resolve(config["ansys_executable"]).resolve()),
        "ansys_executable_sha256": sha256(
            resolve(config["ansys_executable"])
        ),
        "input_sha256": input_hashes,
        "frozen_inputs": copied,
        "free_memory_gib": memory_available_gib(),
        "aedt_processes": aedt_processes(),
        "disk_free_gib": shutil.disk_usage(root).free / (1024.0**3),
        "preregistration_sha256": sha256(preregistration_path),
    }
    write_json(root / "baseline_audit.json", baseline_audit)
    preparation = _prepare_nominal_from_project(
        root,
        preregistration,
        source_project,
        {
            "source_run": str(source_run_root),
            "source_manifest_sha256": source_manifest_audit[
                "manifest_sha256"
            ],
            "source_build_gate_sha256": sha256(build_gate_path),
            "source_build_project_sha256": source_project_sha256,
        },
    )
    continuation_audit = {
        "schema": "v149_nominal_continuation_audit_v1",
        "source_run": str(source_run_root),
        "output_root": str(root.resolve()),
        "source_manifest": source_manifest_audit,
        "source_build_project_sha256": source_project_sha256,
        "source_model_inventory_sha256": inventory_sha256,
        "source_model_inventory": inventory,
        "copied_project_sha256": sha256(Path(preparation["project_path"])),
        "source_run_unchanged": True,
        "solve_started": False,
        "state_scope": "continuation_creation_time",
    }
    write_json(root / "continuation_audit.json", continuation_audit)
    decision = {
        "stage": "A_nominal_periodic_continuation_prepared",
        "allow_periodic_build_smoke": False,
        "allow_periodic_nominal_solve": True,
        "allow_periodic_doe_batch": False,
        "allow_1x1": False,
        "allow_2x2": False,
        "allow_4x4": False,
        "allow_16x16": False,
        "allow_eep_export": False,
        "allow_training_labels": False,
        "allow_critic_training": False,
        "reason": (
            "A verified finalized build project was copied into a new run. "
            "Only one nominal periodic sweep is authorized after resource "
            "preflight."
        ),
    }
    write_json(root / "stage_decision.json", decision)
    authorization = seal_nominal_authorization(
        root,
        Path(preparation["manifest_path"]),
        "finalized_build_continuation",
        {
            "source_run": str(source_run_root),
            "source_manifest_sha256": source_manifest_audit[
                "manifest_sha256"
            ],
            "source_build_gate_sha256": sha256(build_gate_path),
            "source_build_project_sha256": source_project_sha256,
            "source_model_inventory_sha256": inventory_sha256,
        },
    )
    return {
        "output_root": str(root.resolve()),
        "source_run": str(source_run_root),
        "continuation_audit": continuation_audit,
        "preparation": preparation,
        "authorization": authorization,
        "decision": decision,
    }


def prepare_periodic_build_smoke(
    run_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    validate_config(config)
    folder = run_root / "periodic" / "build_smoke"
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite build smoke: {folder}")
    folder.mkdir(parents=True)
    project = folder / "v149_periodic_build_smoke.aedt"
    builder = folder / "build.vbs"
    inventory = folder / "model_inventory.txt"
    validation = folder / "design_validation.txt"
    state = {
        "frequency_ghz": 10.0,
        "theta_deg": 0.0,
        "phi_deg": 0.0,
    }
    source = periodic_builder_text(
        project, config, config["nominal_geometry"], state
    )
    builder.write_text(source, encoding="ascii")
    manifest = {
        "case_id": "periodic_nominal_build_smoke",
        "model_scope": "periodic_unit_only",
        "authorizes_real_solve": False,
        "project_path": str(project.resolve()),
        "model_inventory_path": str(inventory.resolve()),
        "design_validation_path": str(validation.resolve()),
        "builder_path": str(builder.resolve()),
        "builder_sha256": sha256(builder),
        "scan_state": state,
        "geometry": config["nominal_geometry"],
        "geometry_audit": geometry_audit(
            config, config["nominal_geometry"]
        ),
        "expected_named_features": [
            "FeedPort",
            "PrimaryX",
            "SecondaryX",
            "PrimaryY",
            "SecondaryY",
            "PrimaryX_Main",
            "SecondaryX_Main",
            "PrimaryX_Stack",
            "SecondaryX_Stack",
            "PrimaryY_Main",
            "SecondaryY_Main",
            "PrimaryY_Stack",
            "SecondaryY_Stack",
            "FloquetTop",
            "Mesh_ProbeLaunch",
            "Mesh_PortSheet",
            "Mesh_PatchEdges",
            "Mesh_SIWVias",
        ],
    }
    manifest_path = folder / "case_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "folder": str(folder.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "builder_path": str(builder.resolve()),
        "project_path": str(project.resolve()),
    }


def build_smoke_preflight(
    config: dict[str, Any],
    free_memory_gib: float,
    aedt_instance_count: int,
) -> dict[str, Any]:
    minimum = float(config["resources"]["minimum_free_memory_before_build_gib"])
    maximum = int(
        config["resources"]["maximum_concurrent_aedt_instances"]
    )
    if not math.isfinite(free_memory_gib) or free_memory_gib < minimum:
        return {
            "allowed": False,
            "reason": (
                f"Available memory {free_memory_gib:.2f} GiB is below the "
                f"{minimum:.2f} GiB AEDT launch threshold."
            ),
        }
    if aedt_instance_count >= maximum:
        return {
            "allowed": False,
            "reason": (
                f"{aedt_instance_count} AEDT/HFSS instance(s) already exist; "
                "serial execution requires zero."
            ),
        }
    return {"allowed": True, "reason": "Memory and serial-process gates pass."}


def _run_with_memory_guard(
    command: list[str],
    log_path: Path,
    abort_below_gib: float,
    poll_seconds: float,
) -> tuple[int, bool, float]:
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with log_path.open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )
        minimum = memory_available_gib()
        aborted = False
        while process.poll() is None:
            current = memory_available_gib()
            if math.isfinite(current):
                minimum = min(minimum, current)
                if current < abort_below_gib:
                    if os.name == "nt":
                        subprocess.run(
                            [
                                "taskkill",
                                "/PID",
                                str(process.pid),
                                "/T",
                                "/F",
                            ],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                        )
                    else:
                        process.terminate()
                    aborted = True
                    break
            time.sleep(max(0.2, poll_seconds))
        try:
            return_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait()
            aborted = True
    return int(return_code), aborted, minimum


def run_periodic_build_smoke(
    run_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    folder = run_root / "periodic" / "build_smoke"
    manifest = json.loads(
        (folder / "case_manifest.json").read_text(encoding="utf-8")
    )
    audit_path = folder / "run_audit.json"
    if audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite build audit: {audit_path}")
    project_path = Path(manifest["project_path"])
    log_path = folder / "build.log"
    if project_path.exists() or log_path.exists():
        raise FileExistsError(
            "Build project or log already exists without a final audit"
        )
    with aedt_execution_lock():
        processes = aedt_processes()
        free = memory_available_gib()
        preflight = build_smoke_preflight(config, free, len(processes))
        if not preflight["allowed"]:
            audit = {
                "case_id": manifest["case_id"],
                "started": False,
                "blocked": True,
                "preflight": preflight,
                "free_memory_gib_before": free,
                "aedt_processes": processes,
                "project_exists": project_path.exists(),
            }
            write_json(audit_path, audit)
            return audit
        started = time.time()
        return_code, aborted, minimum = _run_with_memory_guard(
            [
                str(resolve(config["ansys_executable"])),
                "-RunScriptAndExit",
                manifest["builder_path"],
            ],
            log_path,
            float(config["resources"]["abort_free_memory_during_solve_gib"]),
            float(config["resources"]["poll_interval_seconds"]),
        )
    audit = {
        "case_id": manifest["case_id"],
        "started": True,
        "blocked": False,
        "preflight": preflight,
        "return_code": return_code,
        "memory_aborted": aborted,
        "free_memory_gib_before": free,
        "minimum_free_memory_gib": minimum,
        "elapsed_seconds": time.time() - started,
        "project_exists": project_path.exists(),
    }
    write_json(audit_path, audit)
    return audit


def validate_model_inventory(
    lines: list[str],
    config: dict[str, Any],
    geometry: dict[str, Any],
) -> dict[str, Any]:
    del config
    objects: set[str] = set()
    boundaries: set[str] = set()
    excitations: list[tuple[str, str]] = []
    for raw in lines:
        parts = raw.strip().split("|")
        if len(parts) == 2 and parts[0] == "OBJECT":
            objects.add(parts[1])
        elif len(parts) == 2 and parts[0] == "BOUNDARY":
            boundaries.add(parts[1])
        elif len(parts) >= 3 and parts[0] == "EXCITATION":
            excitations.append((parts[1], parts[2]))
    required_objects = {
        "MainSubstrate",
        "StackSpacer",
        "Ground",
        "SIWCavityTop",
        "DrivenPatch",
        "StackedPatch",
        "FeedProbe",
        "CoaxOuter",
        "CoaxDielectric",
        "PortSheet",
        "AirCell",
        *{
            f"SIWVia_{index:03d}"
            for index in range(len(siw_via_centers(geometry)))
        },
    }
    required_boundaries = {
        "CopperSheetFiniteConductivity",
        "PrimaryX",
        "SecondaryX",
        "PrimaryY",
        "SecondaryY",
        "PrimaryX_Main",
        "SecondaryX_Main",
        "PrimaryX_Stack",
        "SecondaryX_Stack",
        "PrimaryY_Main",
        "SecondaryY_Main",
        "PrimaryY_Stack",
        "SecondaryY_Stack",
    }
    required_excitations = {"FeedPort", "FloquetTop"}
    excitation_bases = {
        name.split(":", 1)[0] for name, _ in excitations
    }
    checks = {
        "objects": required_objects.issubset(objects),
        "boundaries": required_boundaries.issubset(boundaries),
        "excitations": required_excitations.issubset(excitation_bases),
        "single_feed_mode": sum(
            name.split(":", 1)[0] == "FeedPort"
            for name, _ in excitations
        )
        == 1,
        "two_floquet_modes": sum(
            name.split(":", 1)[0] == "FloquetTop"
            for name, _ in excitations
        )
        == 2,
    }
    return {
        "verified": all(checks.values()),
        "checks": checks,
        "missing_objects": sorted(required_objects - objects),
        "missing_boundaries": sorted(required_boundaries - boundaries),
        "missing_excitations": sorted(
            required_excitations - excitation_bases
        ),
        "object_count": len(objects),
        "boundary_count": len(boundaries),
        "excitation_mode_count": len(excitations),
    }


def critical_log_hits(text: str) -> dict[str, int]:
    lowered = text.lower()
    terms = (
        "[error]",
        "script error",
        "small segment",
        "port assignment failed",
        "boundary assignment failed",
        "geometry error",
        "body could not be created",
        "conductors touch lumped port",
        "intersect",
    )
    return {term: lowered.count(term) for term in terms if term in lowered}


def audit_periodic_build_smoke(
    run_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    folder = run_root / "periodic" / "build_smoke"
    gate_path = folder / "build_gate.json"
    if gate_path.exists():
        raise FileExistsError(f"Refusing to overwrite build gate: {gate_path}")
    manifest = json.loads(
        (folder / "case_manifest.json").read_text(encoding="utf-8")
    )
    run_path = folder / "run_audit.json"
    run = (
        json.loads(run_path.read_text(encoding="utf-8"))
        if run_path.exists()
        else {"started": False, "blocked": True}
    )
    log_path = folder / "build.log"
    log_text = (
        log_path.read_text(encoding="utf-8", errors="ignore").lower()
        if log_path.exists()
        else ""
    )
    warning_hits = critical_log_hits(log_text)
    source = Path(manifest["builder_path"]).read_text(encoding="ascii")
    named_feature_checks = {
        name: name in source
        for name in manifest["expected_named_features"]
    }
    inventory_path = Path(manifest["model_inventory_path"])
    inventory = (
        validate_model_inventory(
            inventory_path.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines(),
            config,
            manifest["geometry"],
        )
        if inventory_path.exists()
        else {
            "verified": False,
            "checks": {},
            "missing_objects": ["inventory_file"],
            "missing_boundaries": [],
            "missing_excitations": [],
        }
    )
    validation_path = Path(manifest.get("design_validation_path", ""))
    validation_text = (
        validation_path.read_text(
            encoding="utf-8", errors="ignore"
        ).strip()
        if validation_path.is_file()
        else ""
    )
    design_validation_passed = validation_text == "VALIDATION|1"
    passed = bool(
        run.get("started")
        and not run.get("blocked")
        and run.get("return_code") == 0
        and not run.get("memory_aborted")
        and Path(manifest["project_path"]).exists()
        and not warning_hits
        and all(named_feature_checks.values())
        and inventory["verified"]
        and design_validation_passed
    )
    result = {
        "evidence_source": "HFSS_build_only",
        "build_smoke_passed": passed,
        "does_not_authorize_physical_metrics": True,
        "run_audit": run,
        "critical_warning_hits": warning_hits,
        "named_feature_checks": named_feature_checks,
        "model_inventory_verified": inventory["verified"],
        "model_inventory": inventory,
        "design_validation_passed": design_validation_passed,
        "design_validation_text": validation_text,
        "project_exists": Path(manifest["project_path"]).exists(),
        "project_sha256": (
            sha256(Path(manifest["project_path"]))
            if Path(manifest["project_path"]).exists()
            else None
        ),
    }
    write_json(gate_path, result)
    decision_path = run_root / "stage_decision.json"
    if decision_path.exists():
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision.update(
            {
                "stage": "A_periodic_build_gate_evaluated",
                "periodic_build_smoke_passed": passed,
                "allow_periodic_nominal_solve": passed,
                "allow_periodic_doe_batch": False,
                "allow_1x1": False,
                "reason": (
                    "Periodic native CAD build passed; only one nominal periodic full-wave solve may proceed after a fresh resource preflight."
                    if passed
                    else "Periodic build evidence is absent, blocked, or failed; every physical solve and later stage remains locked."
                ),
            }
        )
        write_json(decision_path, decision)
    return result


def _prepare_nominal_from_project(
    run_root: Path,
    config: dict[str, Any],
    source_project: Path,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    folder = run_root / "periodic" / "nominal_solve"
    if folder.exists():
        raise FileExistsError(
            f"Refusing to overwrite nominal solve preparation: {folder}"
        )
    folder.mkdir(parents=True)
    project = folder / "v149_periodic_nominal_solve.aedt"
    shutil.copy2(source_project, project)
    touchstone = folder / "v149_periodic_nominal.s3p"
    source_names = folder / "source_names.txt"
    solver = folder / "solve_export.vbs"
    solver.write_text(
        periodic_solver_text(
            project, touchstone, source_names, config
        ),
        encoding="ascii",
    )
    manifest = {
        "case_id": "periodic_nominal_broadside",
        "model_scope": "periodic_unit_only",
        "scan_state": {
            "theta_deg": 0.0,
            "phi_deg": 0.0,
            "frequency_sweep_ghz": config["sweep_ghz"],
        },
        "project_path": str(project.resolve()),
        "project_sha256_before_solve": sha256(project),
        "solver_path": str(solver.resolve()),
        "solver_sha256": sha256(solver),
        "touchstone_path": str(touchstone.resolve()),
        "source_names_path": str(source_names.resolve()),
        "solve_started": False,
        "minimum_free_memory_before_solve_gib": float(
            config["resources"]["minimum_free_memory_before_solve_gib"]
        ),
        "authorizes_doe_batch": False,
        "authorizes_1x1": False,
    }
    if provenance:
        manifest.update(provenance)
    write_json(folder / "case_manifest.json", manifest)
    return {
        "folder": str(folder.resolve()),
        "project_path": str(project.resolve()),
        "solver_path": str(solver.resolve()),
        "manifest_path": str((folder / "case_manifest.json").resolve()),
        "solve_started": False,
    }


def prepare_nominal_periodic_solve(
    run_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    build_folder = run_root / "periodic" / "build_smoke"
    build_gate = json.loads(
        (build_folder / "build_gate.json").read_text(encoding="utf-8")
    )
    if build_gate.get("build_smoke_passed") is not True:
        raise RuntimeError(
            "Nominal solve remains locked until the HFSS build smoke passes"
        )
    if build_gate.get("model_inventory_verified") is not True:
        raise RuntimeError("Saved AEDT model inventory was not verified")
    build_manifest = json.loads(
        (build_folder / "case_manifest.json").read_text(encoding="utf-8")
    )
    source_project = Path(build_manifest["project_path"])
    if not source_project.exists():
        raise FileNotFoundError(source_project)
    if build_gate.get("project_sha256") != sha256(source_project):
        raise RuntimeError("Built AEDT project hash changed after audit")
    return _prepare_nominal_from_project(run_root, config, source_project)


def validate_nominal_authorization(
    run_root: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    run_root = run_root.resolve()
    authorization_path = run_root / "solve_authorization.json"
    if not authorization_path.is_file():
        raise RuntimeError("Nominal solve authorization is missing")
    authorization = json.loads(
        authorization_path.read_text(encoding="utf-8")
    )
    if authorization.get("schema") != "v149_nominal_solve_authorization_v1":
        raise RuntimeError("Nominal solve authorization schema is invalid")
    if authorization.get("issued_for_run") != str(run_root):
        raise RuntimeError("Nominal solve authorization targets another run")
    static_files = {
        "case_manifest_sha256": (
            run_root / "periodic" / "nominal_solve" / "case_manifest.json"
        ),
        "stage_decision_sha256": run_root / "stage_decision.json",
        "baseline_audit_sha256": run_root / "baseline_audit.json",
        "preregistration_sha256": run_root / "preregistration.json",
    }
    continuation_path = run_root / "continuation_audit.json"
    if authorization.get("continuation_audit_sha256") is not None:
        static_files["continuation_audit_sha256"] = continuation_path
    for field, path in static_files.items():
        if not path.is_file() or sha256(path) != authorization.get(field):
            raise RuntimeError(
                f"Nominal solve authorization hash mismatch: {path.name}"
            )

    decision = json.loads(
        (run_root / "stage_decision.json").read_text(encoding="utf-8")
    )
    forbidden = (
        "allow_periodic_build_smoke",
        "allow_periodic_doe_batch",
        "allow_1x1",
        "allow_2x2",
        "allow_4x4",
        "allow_16x16",
        "allow_eep_export",
        "allow_training_labels",
        "allow_critic_training",
    )
    if decision.get("allow_periodic_nominal_solve") is not True or any(
        decision.get(name) is not False for name in forbidden
    ):
        raise RuntimeError("Nominal solve authorization stage lock is invalid")

    snapshot = _path_within(
        Path(authorization["control_snapshot_path"]),
        run_root,
        "Control-script snapshot",
    )
    snapshot_hash = sha256(snapshot) if snapshot.is_file() else None
    if snapshot_hash != authorization.get("control_snapshot_sha256"):
        raise RuntimeError("Nominal solve authorization snapshot mismatch")
    if snapshot_hash != sha256(Path(__file__).resolve()):
        raise RuntimeError(
            "Current orchestration script differs from the authorized snapshot"
        )

    nominal_root = run_root / "periodic" / "nominal_solve"
    project = _path_within(
        Path(manifest["project_path"]), nominal_root, "Nominal project"
    )
    solver = _path_within(
        Path(manifest["solver_path"]), nominal_root, "Nominal solver"
    )
    touchstone = _path_within(
        Path(manifest["touchstone_path"]), nominal_root, "Touchstone output"
    )
    source_names = _path_within(
        Path(manifest["source_names_path"]), nominal_root, "Source-name output"
    )
    if not project.is_file() or not solver.is_file():
        raise RuntimeError("Nominal solve authorization input is missing")
    if sha256(project) != manifest.get("project_sha256_before_solve"):
        raise RuntimeError("Nominal solve authorization project mismatch")
    if sha256(solver) != manifest.get("solver_sha256"):
        raise RuntimeError("Nominal solve authorization solver mismatch")
    config = load_run_config(run_root)
    baseline = json.loads(
        (run_root / "baseline_audit.json").read_text(encoding="utf-8")
    )
    executable = resolve(config["ansys_executable"])
    if not executable.is_file() or sha256(executable) != baseline.get(
        "ansys_executable_sha256"
    ):
        raise RuntimeError(
            "Authorized AEDT executable is missing or its SHA-256 changed"
        )
    expected_solver = periodic_solver_text(
        project, touchstone, source_names, config
    )
    if solver.read_text(encoding="ascii") != expected_solver:
        raise RuntimeError(
            "Nominal solve script is not the deterministic frozen generator output"
        )

    if authorization.get("authorization_kind") == "finalized_build_continuation":
        source = Path(
            authorization["source_evidence"]["source_run"]
        ).resolve()
        source_manifest = verify_finalized_run_manifest(source)
        evidence = authorization["source_evidence"]
        if source_manifest["manifest_sha256"] != evidence.get(
            "source_manifest_sha256"
        ):
            raise RuntimeError("Nominal solve source manifest changed")
        build = source / "periodic" / "build_smoke"
        build_gate_path = build / "build_gate.json"
        build_manifest = json.loads(
            (build / "case_manifest.json").read_text(encoding="utf-8")
        )
        source_project = _path_within(
            Path(build_manifest["project_path"]),
            source,
            "Authorized source project",
        )
        source_inventory = _path_within(
            Path(build_manifest["model_inventory_path"]),
            source,
            "Authorized model inventory",
        )
        if sha256(build_gate_path) != evidence.get("source_build_gate_sha256"):
            raise RuntimeError("Nominal solve source build gate changed")
        if sha256(source_project) != evidence.get(
            "source_build_project_sha256"
        ) or sha256(project) != sha256(source_project):
            raise RuntimeError("Nominal solve source project changed")
        inventory = validate_model_inventory(
            source_inventory.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines(),
            config,
            config["nominal_geometry"],
        )
        if inventory.get("verified") is not True or sha256(
            source_inventory
        ) != evidence.get("source_model_inventory_sha256"):
            raise RuntimeError("Nominal solve source inventory changed")
    else:
        raise RuntimeError("Unsupported nominal solve authorization kind")
    return {
        "authorization_path": str(authorization_path.resolve()),
        "authorization_sha256": sha256(authorization_path),
        "authorization_kind": authorization["authorization_kind"],
    }


def consume_nominal_launch_authorization(
    run_root: Path,
    manifest: dict[str, Any],
    authorization: dict[str, Any],
) -> Path:
    path = run_root / "periodic" / "nominal_solve" / "launch_intent.json"
    payload = {
        "schema": "v149_nominal_launch_intent_v1",
        "case_id": manifest["case_id"],
        "authorization_sha256": authorization["authorization_sha256"],
        "consumed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": os.getpid(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="ascii") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
    except FileExistsError as error:
        raise RuntimeError(
            "Nominal solve authorization was already consumed"
        ) from error
    return path


def run_nominal_periodic_solve(
    run_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    if (run_root / "stage_summary.json").exists() or (
        run_root / "sha256_manifest.csv"
    ).exists():
        raise RuntimeError(
            "Run is finalized and immutable; allocate a continuation run "
            "before executing the nominal solve"
        )
    folder = run_root / "periodic" / "nominal_solve"
    manifest_path = folder / "case_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    authorization = validate_nominal_authorization(run_root, manifest)
    audit_path = folder / "run_audit.json"
    if audit_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite nominal solve audit: {audit_path}"
        )
    if (folder / "solve_export.log").exists():
        raise FileExistsError("Nominal solve log already exists")
    if (folder / "launch_intent.json").exists():
        raise RuntimeError("Nominal solve authorization was already consumed")
    project_path = Path(manifest["project_path"])
    solver_path = Path(manifest["solver_path"])
    if sha256(project_path) != manifest.get("project_sha256_before_solve"):
        raise RuntimeError("Nominal AEDT project changed after preparation")
    if sha256(solver_path) != manifest.get("solver_sha256"):
        raise RuntimeError("Nominal solve script changed after preparation")
    with aedt_execution_lock():
        processes = aedt_processes()
        free = memory_available_gib()
        preflight = solve_preflight(config, free, len(processes))
        if not preflight["allowed"]:
            blocked = {
                "case_id": manifest["case_id"],
                "started": False,
                "blocked": True,
                "preflight": preflight,
                "free_memory_gib": free,
                "aedt_processes": processes,
            }
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            write_json(folder / f"preflight_block_{timestamp}.json", blocked)
            return blocked
        launch_intent = consume_nominal_launch_authorization(
            run_root, manifest, authorization
        )
        started = time.time()
        return_code, aborted, minimum = _run_with_memory_guard(
            [
                str(resolve(config["ansys_executable"])),
                "-ng",
                "-RunScriptAndExit",
                manifest["solver_path"],
            ],
            folder / "solve_export.log",
            float(config["resources"]["abort_free_memory_during_solve_gib"]),
            float(config["resources"]["poll_interval_seconds"]),
        )
    audit = {
        "schema": "v149_nominal_periodic_run_audit_v1",
        "aedt_version": "2023.1",
        "ansys_executable_sha256": sha256(
            resolve(config["ansys_executable"])
        ),
        "case_id": manifest["case_id"],
        "started": True,
        "blocked": False,
        "preflight": preflight,
        "return_code": return_code,
        "memory_aborted": aborted,
        "free_memory_gib_before": free,
        "minimum_free_memory_gib": minimum,
        "elapsed_seconds": time.time() - started,
        "touchstone_exists": Path(manifest["touchstone_path"]).exists(),
        "source_names_exist": Path(manifest["source_names_path"]).exists(),
        "project_sha256_after_solve": (
            sha256(Path(manifest["project_path"]))
            if Path(manifest["project_path"]).exists()
            else None
        ),
        "touchstone_sha256": (
            sha256(Path(manifest["touchstone_path"]))
            if Path(manifest["touchstone_path"]).exists()
            else None
        ),
        "source_names_sha256": (
            sha256(Path(manifest["source_names_path"]))
            if Path(manifest["source_names_path"]).exists()
            else None
        ),
        "solver_sha256_executed": sha256(solver_path),
        "solver_log_sha256": sha256(folder / "solve_export.log"),
        "authorization_sha256": authorization["authorization_sha256"],
        "launch_intent_sha256": sha256(launch_intent),
        "convergence_artifacts": convergence_artifact_manifest(folder),
        "result_directory": str(folder.resolve()),
    }
    write_json(audit_path, audit)
    return audit


def parse_touchstone(
    path: Path, nports: int
) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    """Parse an HFSS Touchstone v1 export into frequency-major S matrices."""
    unit = "ghz"
    data_format = "ma"
    reference_ohm = 50.0
    tokens: list[float] = []
    named_ports: dict[int, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        port_match = re.match(
            r"\s*!\s*Port\[(\d+)\]\s*=\s*(.+?)\s*$", raw,
            flags=re.IGNORECASE,
        )
        if port_match:
            named_ports[int(port_match.group(1)) - 1] = port_match.group(2)
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line[1:].lower().split()
            if len(parts) < 3 or "s" not in parts:
                raise ValueError(f"Unsupported Touchstone option line: {line}")
            unit = parts[0]
            data_format = parts[2]
            if "r" in parts:
                reference_ohm = float(parts[parts.index("r") + 1])
            continue
        if line.startswith("["):
            continue
        tokens.extend(float(value) for value in line.split())
    record_length = 1 + 2 * nports * nports
    if not tokens or len(tokens) % record_length:
        raise ValueError(f"Invalid Touchstone token count in {path}")
    raw = np.asarray(tokens, dtype=float).reshape(-1, record_length)
    unit_scale = {
        "hz": 1.0e-9,
        "khz": 1.0e-6,
        "mhz": 1.0e-3,
        "ghz": 1.0,
    }
    if unit not in unit_scale:
        raise ValueError(f"Unsupported Touchstone frequency unit: {unit}")
    frequencies_ghz = raw[:, 0] * unit_scale[unit]
    pairs = raw[:, 1:].reshape(-1, nports, nports, 2)
    if data_format == "ma":
        values = pairs[..., 0] * np.exp(
            1j * np.deg2rad(pairs[..., 1])
        )
    elif data_format == "ri":
        values = pairs[..., 0] + 1j * pairs[..., 1]
    elif data_format == "db":
        values = 10.0 ** (pairs[..., 0] / 20.0) * np.exp(
            1j * np.deg2rad(pairs[..., 1])
        )
    else:
        raise ValueError(f"Unsupported Touchstone data format: {data_format}")
    # Touchstone v1 lists S11,S21,... by input-port column.
    matrices = np.transpose(values, (0, 2, 1))
    port_names = (
        [named_ports[index] for index in range(nports)]
        if set(named_ports) == set(range(nports))
        else []
    )
    return frequencies_ghz, matrices, reference_ohm, port_names


def bind_exported_port_modes(
    source_names: list[str], touchstone_ports: list[str]
) -> dict[str, list[int]]:
    """Bind HFSS boundary sources to modal Touchstone ports."""
    source_bases = {name.split(":", 1)[0] for name in source_names}
    touchstone_bases = {
        name.split(":", 1)[0] for name in touchstone_ports
    }
    if source_bases != touchstone_bases:
        raise RuntimeError(
            "Touchstone modal ports do not match GetAllSources() boundaries"
        )
    feed_indices = [
        index
        for index, name in enumerate(touchstone_ports)
        if name.split(":", 1)[0] == "FeedPort"
    ]
    floquet_indices = [
        index
        for index, name in enumerate(touchstone_ports)
        if name.split(":", 1)[0] == "FloquetTop"
    ]
    if (
        len(touchstone_ports) != 3
        or len(feed_indices) != 1
        or len(floquet_indices) != 2
    ):
        raise RuntimeError(
            "Expected exactly one feed mode and two Floquet modes in S3P"
        )
    return {
        "feed_indices": feed_indices,
        "floquet_indices": floquet_indices,
    }


def convergence_profile_metrics(folder: Path) -> dict[str, Any]:
    values: list[float] = []
    maximum_tetrahedra = 0
    peak_solver_memory_kb = 0
    converged = False
    small_segment_count = 0
    for path in folder.rglob("*.profile"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        matrix_sections_present = (
            "Matrix Assembly/Solve" in text
            or (
                "ProfileItem('Matrix Assembly'" in text
                and "ProfileItem('Matrix Solve'" in text
            )
        )
        authentic = matrix_sections_present and all(
            token in text
            for token in (
                "$begin 'Profile'",
                "HFSS Version 2023.1.0",
                "HFSSCOMENGINE.exe",
                "Max Mag. Delta S",
                "Status\\', \\'Normal Completion",
            )
        ) and "Engine Detected Error" not in text
        converged = converged or authentic
        for line in text.splitlines():
            if "Max Mag. Delta S" in line:
                match = re.search(
                    r"Max Mag\. Delta S[^0-9+\-.]*([0-9.eE+\-]+)",
                    line,
                )
                if match:
                    values.append(float(match.group(1)))
            if "Tetrahedra" in line and "ProfileItem" in line:
                matches = re.findall(
                    r"Tetrahedra\\?',\s*(\d+)", line
                )
                if matches:
                    maximum_tetrahedra = max(
                        maximum_tetrahedra, max(map(int, matches))
                    )
            if "ProfileItem('Matrix Solve'" in line:
                match = re.search(r",\s*(\d+),\s*'I\(", line)
                if match:
                    peak_solver_memory_kb = max(
                        peak_solver_memory_kb, int(match.group(1))
                    )
    for path in folder.rglob("*.g3derr"):
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        small_segment_count += len(
            re.findall(
                r"^\s*26\s+small mesh segment detected", text, re.MULTILINE
            )
        )
    return {
        "pass_count": len(values),
        "final_delta_s": values[-1] if values else None,
        "converged": converged,
        "small_segment_count": small_segment_count,
        "maximum_tetrahedra": maximum_tetrahedra,
        "peak_solver_memory_gib": peak_solver_memory_kb / 1024.0**2,
    }


def convergence_artifact_manifest(folder: Path) -> list[dict[str, Any]]:
    artifacts = sorted(
        {
            *folder.rglob("*.profile"),
            *folder.rglob("*.g3derr"),
        }
    )
    return [
        {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in artifacts
        if path.is_file()
    ]


def verify_convergence_artifacts(
    folder: Path, recorded: Any
) -> bool:
    if not isinstance(recorded, list) or not recorded:
        return False
    current = convergence_artifact_manifest(folder)
    if len(current) != len(recorded):
        return False
    expected_by_path = {
        str(Path(item["path"]).resolve()): item for item in recorded
    }
    current_by_path = {item["path"]: item for item in current}
    if set(expected_by_path) != set(current_by_path):
        return False
    return all(
        _valid_sha256(expected_by_path[path].get("sha256"))
        and expected_by_path[path].get("sha256") == item["sha256"]
        and int(expected_by_path[path].get("size_bytes", -1))
        == int(item["size_bytes"])
        for path, item in current_by_path.items()
    )


def analyze_nominal_periodic_exports(
    manifest: dict[str, Any],
    audit: dict[str, Any],
    config: dict[str, Any],
    solver_log: Path,
) -> dict[str, Any]:
    """Audit the frozen broadside export without opening the 45-state gate."""
    if not (
        audit.get("started") is True
        and audit.get("blocked") is False
        and audit.get("return_code") == 0
        and audit.get("memory_aborted") is False
    ):
        raise RuntimeError("Nominal HFSS solve did not complete successfully")
    ansys_executable = resolve(config["ansys_executable"])
    if not (
        audit.get("schema") == "v149_nominal_periodic_run_audit_v1"
        and audit.get("aedt_version") == "2023.1"
        and ansys_executable.is_file()
        and audit.get("ansys_executable_sha256")
        == sha256(ansys_executable)
    ):
        raise RuntimeError("Nominal solve lacks an independent AEDT run audit")
    paths = {
        "project": Path(manifest["project_path"]),
        "touchstone": Path(manifest["touchstone_path"]),
        "source_names": Path(manifest["source_names_path"]),
        "solver_log": solver_log,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    nominal_state = (
        float(config["frequencies_ghz"][1]),
        float(manifest["scan_state"]["theta_deg"]),
        float(manifest["scan_state"]["phi_deg"]),
    )
    if not validate_aedt_project_container(paths["project"]):
        raise RuntimeError("Nominal project lacks a saved LastAdaptive HFSS solution")
    if not aedt_project_scan_state_matches(paths["project"], nominal_state):
        raise RuntimeError("Nominal project scan state is not broadside 10 GHz")
    expected_hashes = {
        "project": audit.get("project_sha256_after_solve"),
        "touchstone": audit.get("touchstone_sha256"),
        "source_names": audit.get("source_names_sha256"),
        "solver_log": audit.get("solver_log_sha256"),
    }
    for name in ("project", "touchstone", "source_names", "solver_log"):
        if not _valid_sha256(expected_hashes[name]):
            raise RuntimeError(f"Missing valid post-solve {name} hash")
        if expected_hashes[name] != sha256(paths[name]):
            raise RuntimeError(f"Post-solve {name} hash mismatch")
    if (
        not _valid_sha256(audit.get("solver_sha256_executed"))
        or audit.get("solver_sha256_executed")
        != manifest.get("solver_sha256")
    ):
        raise RuntimeError("Executed solver hash does not match the manifest")
    if not verify_convergence_artifacts(
        paths["project"].parent, audit.get("convergence_artifacts")
    ):
        raise RuntimeError("Convergence artifact set or hash changed")
    if Path(audit.get("result_directory", "")).resolve() != paths[
        "project"
    ].parent.resolve():
        raise RuntimeError("Nominal run-audit result directory mismatch")
    source_names = [
        line.strip()
        for line in paths["source_names"].read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines()
        if line.strip()
    ]
    frequencies, matrices, reference_ohm, touchstone_ports = parse_touchstone(
        paths["touchstone"], 3
    )
    if not touchstone_ports:
        raise RuntimeError("Touchstone export lacks Port[n] mode comments")
    binding = bind_exported_port_modes(source_names, touchstone_ports)
    feed = binding["feed_indices"][0]
    floquet_indices = binding["floquet_indices"]
    rows: list[dict[str, Any]] = []
    tolerance_ghz = 5.0e-5
    for requested in map(float, config["frequencies_ghz"]):
        index = int(np.argmin(np.abs(frequencies - requested)))
        actual = float(frequencies[index])
        if abs(actual - requested) > tolerance_ghz:
            raise RuntimeError(
                f"Touchstone lacks required frequency {requested:.5f} GHz"
            )
        matrix = matrices[index]
        gamma = complex(matrix[feed, feed])
        magnitude = abs(gamma)
        denominator = 1.0 - gamma
        impedance = (
            reference_ohm * (1.0 + gamma) / denominator
            if abs(denominator) > 1.0e-12
            else complex(math.inf, math.inf)
        )
        reflected_power = magnitude**2
        accepted_power = 1.0 - reflected_power
        floquet_power = float(
            sum(abs(matrix[index_, feed]) ** 2 for index_ in floquet_indices)
        )
        accepted_efficiency = (
            floquet_power / accepted_power
            if accepted_power > 1.0e-12
            else math.nan
        )
        rows.append(
            {
                "frequency_ghz": requested,
                "actual_frequency_ghz": actual,
                "theta_deg": float(manifest["scan_state"]["theta_deg"]),
                "phi_deg": float(manifest["scan_state"]["phi_deg"]),
                "evidence_source": "HFSS_periodic_nominal_broadside",
                "solve_complete": True,
                "provenance_verified": True,
                "project_sha256": expected_hashes["project"],
                "touchstone_sha256": expected_hashes["touchstone"],
                "solver_log_sha256": expected_hashes["solver_log"],
                "source_names_sha256": expected_hashes["source_names"],
                "active_rl_db": -20.0 * math.log10(max(magnitude, 1.0e-15)),
                "passive_rl_db": -20.0 * math.log10(max(magnitude, 1.0e-15)),
                "input_impedance_real_ohm": float(impedance.real),
                "input_impedance_imag_ohm": float(impedance.imag),
                "reflection_power": reflected_power,
                "accepted_power": accepted_power,
                "floquet_power": floquet_power,
                "accepted_power_efficiency": accepted_efficiency,
                "system_power_efficiency": floquet_power,
            }
        )
    profile = convergence_profile_metrics(paths["project"].parent)
    log_hits = critical_log_hits(
        paths["solver_log"].read_text(encoding="utf-8", errors="ignore")
    )
    power_consistent = all(
        math.isfinite(float(row["accepted_power_efficiency"]))
        and -1.0e-6 <= float(row["reflection_power"]) <= 1.0 + 1.0e-6
        and -1.0e-6 <= float(row["floquet_power"])
        and float(row["floquet_power"])
        <= float(row["accepted_power"]) + 1.0e-3
        and float(row["accepted_power_efficiency"]) <= 1.001
        for row in rows
    )
    convergence_complete = bool(
        profile["converged"]
        and isinstance(profile["final_delta_s"], (int, float))
        and math.isfinite(float(profile["final_delta_s"]))
        and float(profile["final_delta_s"])
        <= float(config["gates"]["maximum_final_delta_s"])
    )
    evidence_complete = bool(
        power_consistent
        and convergence_complete
        and not log_hits
        and profile["small_segment_count"] == 0
    )
    return {
        "case_id": manifest["case_id"],
        "evidence_source": "HFSS_periodic_nominal_broadside",
        "nominal_export_evidence_complete": evidence_complete,
        "power_consistency_passed": power_consistent,
        "convergence_evidence_complete": convergence_complete,
        "authorizes_periodic_gate": False,
        "authorizes_doe_batch": False,
        "authorizes_1x1": False,
        "reason": (
            "Three-frequency broadside export is diagnostic only; all 45 "
            "preregistered frequency-angle states and scan-gain evidence are "
            "still required."
        ),
        "source_names": source_names,
        "touchstone_port_order": touchstone_ports,
        "profile": profile,
        "critical_warning_hits": log_hits,
        "rows": rows,
    }


def analyze_nominal_periodic_solve(
    run_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    del config
    folder = run_root / "periodic" / "nominal_solve"
    result_path = folder / "nominal_analysis.json"
    csv_path = folder / "nominal_three_frequency_metrics.csv"
    if result_path.exists() or csv_path.exists():
        raise FileExistsError("Refusing to overwrite nominal analysis")
    manifest = json.loads(
        (folder / "case_manifest.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (folder / "run_audit.json").read_text(encoding="utf-8")
    )
    result = analyze_nominal_periodic_exports(
        manifest, audit, load_run_config(run_root), folder / "solve_export.log"
    )
    write_csv(csv_path, result["rows"])
    write_json(result_path, result)
    return result


def generate_periodic_doe(
    config: dict[str, Any]
) -> list[dict[str, Any]]:
    validate_config(config)
    ranges = config["manufacturing_ranges"]
    count = int(config["doe"]["sample_count"])
    seed = int(config["doe"]["random_seed"])
    randomizer = random.Random(seed)
    columns: dict[str, list[float]] = {}
    for name, limits in sorted(ranges.items()):
        low, high = map(float, limits)
        values = [
            low + (high - low) * (index + randomizer.random()) / count
            for index in range(count)
        ]
        randomizer.shuffle(values)
        columns[name] = values
    rows: list[dict[str, Any]] = []
    for index in range(count):
        row: dict[str, Any] = {
            "candidate_id": f"doe_{index:02d}",
            "doe_method": "constraint_aware_latin_hypercube",
            "random_seed": seed,
        }
        for name in sorted(columns):
            row[name] = round(columns[name][index], 7)
        geometry = dict(config["nominal_geometry"])
        geometry.update(
            {name: row[name] for name in config["manufacturing_ranges"]}
        )
        validate_geometry(config, geometry)
        rows.append(row)
    return rows


def prepare_doe_manifest(
    run_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    csv_path = run_root / "manifests" / "periodic_doe.csv"
    json_path = run_root / "manifests" / "periodic_doe.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("Refusing to overwrite periodic DOE manifest")
    rows = generate_periodic_doe(config)
    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "protocol": config["protocol"],
            "model_scope": "periodic_unit_only",
            "authorizes_hfss_batch": False,
            "sample_count": len(rows),
            "parameter_count": len(config["manufacturing_ranges"]),
            "samples": rows,
        },
    )
    return {
        "sample_count": len(rows),
        "csv_path": str(csv_path.resolve()),
        "json_path": str(json_path.resolve()),
        "authorizes_hfss_batch": False,
    }


def active_return_metrics(
    s_matrix: np.ndarray,
    excitation: np.ndarray,
    significant_relative_amplitude: float = 0.05,
) -> dict[str, Any]:
    s = np.asarray(s_matrix, dtype=complex)
    a = np.asarray(excitation, dtype=complex).reshape(-1)
    if s.shape != (a.size, a.size):
        raise ValueError("S matrix and excitation dimensions do not agree")
    if not np.any(np.abs(a) > 0.0):
        raise ValueError("Excitation cannot be all zero")
    b = s @ a
    nonzero = np.abs(a) > 1.0e-12
    significant = (
        np.abs(a)
        >= float(significant_relative_amplitude) * float(np.max(np.abs(a)))
    ) & nonzero
    gamma = np.full(a.shape, np.nan + 1j * np.nan, dtype=complex)
    gamma[nonzero] = b[nonzero] / a[nonzero]
    active_rl = np.full(a.shape, np.nan, dtype=float)
    active_rl[nonzero] = -20.0 * np.log10(
        np.maximum(np.abs(gamma[nonzero]), 1.0e-15)
    )
    incident_power = float(np.vdot(a, a).real)
    reflected_power = float(np.vdot(b, b).real)
    total_rl = -10.0 * math.log10(
        max(reflected_power / max(incident_power, 1.0e-30), 1.0e-30)
    )
    return {
        "minimum_significant_active_rl_db": float(
            np.min(active_rl[significant])
        ),
        "minimum_all_nonzero_active_rl_db": float(
            np.min(active_rl[nonzero])
        ),
        "total_rl_db": total_rl,
        "significant_port_count": int(np.count_nonzero(significant)),
        "all_nonzero_port_count": int(np.count_nonzero(nonzero)),
        "incident_power_normalized": incident_power,
        "reflected_power_normalized": reflected_power,
        "active_rl_db_by_port": [
            None if not np.isfinite(value) else float(value)
            for value in active_rl
        ],
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    maximize = ("minimum_active_rl_db", "minimum_efficiency")
    minimize = ("maximum_scan_gain_drop_db", "maximum_final_delta_s")
    no_worse = all(
        float(left.get(name, -math.inf))
        >= float(right.get(name, -math.inf))
        for name in maximize
    ) and all(
        float(left.get(name, math.inf))
        <= float(right.get(name, math.inf))
        for name in minimize
    )
    strictly_better = any(
        float(left.get(name, -math.inf))
        > float(right.get(name, -math.inf))
        for name in maximize
    ) or any(
        float(left.get(name, math.inf))
        < float(right.get(name, math.inf))
        for name in minimize
    )
    return no_worse and strictly_better


def select_pareto_candidates(
    rows: list[dict[str, Any]], maximum_count: int
) -> list[dict[str, Any]]:
    if not 3 <= maximum_count <= 5:
        raise ValueError("v1.49 Pareto selection is limited to 3-5 candidates")
    selected: list[dict[str, Any]] = []
    remaining = list(rows)
    rank = 0
    while remaining and len(selected) < maximum_count:
        front = [
            row
            for row in remaining
            if not any(
                other is not row and _dominates(other, row)
                for other in remaining
            )
        ]
        front.sort(
            key=lambda row: (
                -float(row.get("minimum_active_rl_db", -math.inf)),
                -float(row.get("minimum_efficiency", -math.inf)),
                float(row.get("maximum_scan_gain_drop_db", math.inf)),
                float(row.get("maximum_final_delta_s", math.inf)),
                str(row.get("candidate_id", "")),
            )
        )
        for row in front:
            selected.append({**row, "pareto_rank": rank})
            if len(selected) == maximum_count:
                break
        front_ids = {id(row) for row in front}
        remaining = [row for row in remaining if id(row) not in front_ids]
        rank += 1
    return selected


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _single_convergence_metrics(
    profile_path: Path, geometry_error_path: Path
) -> dict[str, Any]:
    text = profile_path.read_text(encoding="utf-8", errors="ignore")
    values = []
    for line in text.splitlines():
        if "Max Mag. Delta S" not in line:
            continue
        match = re.search(
            r"Max Mag\. Delta S[^0-9+\-.]*([0-9.eE+\-]+)", line
        )
        if match:
            values.append(float(match.group(1)))
    errors = geometry_error_path.read_text(
        encoding="utf-8", errors="ignore"
    ).lower()
    authentic = all(
        token in text
        for token in (
            "$begin 'Profile'",
            "HFSS Version 2023.1.0",
            "HFSSCOMENGINE.exe",
            "Matrix Assembly/Solve",
            "Max Mag. Delta S",
            "Status\\', \\'Normal Completion",
        )
    ) and "Engine Detected Error" not in text
    return {
        "converged": authentic and bool(values),
        "final_delta_s": values[-1] if values else math.nan,
        "small_segment_count": errors.count("small mesh segment"),
    }


def _aedt_named_blocks(text: str, name: str) -> list[str]:
    lines = text.splitlines()
    marker = f"$begin '{name}'"
    blocks: list[str] = []
    for start, line in enumerate(lines):
        if line.strip() != marker:
            continue
        depth = 0
        for stop in range(start, len(lines)):
            stripped = lines[stop].strip()
            if stripped.startswith("$begin '"):
                depth += 1
            elif stripped.startswith("$end '"):
                depth -= 1
                if depth == 0:
                    blocks.append("\n".join(lines[start : stop + 1]))
                    break
    return blocks


def validate_aedt_project_container(
    path: Path, require_solved_adaptive: bool = True
) -> bool:
    if not path.is_file() or path.stat().st_size < 100_000:
        return False
    with path.open("rb") as handle:
        prefix = handle.read(256)
    if not prefix.startswith(b"$begin 'AnsoftProject'"):
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    models = _aedt_named_blocks(text, "HFSSModel")
    if len(models) != 1:
        return False
    model = models[0]
    if "Name='V149_PeriodicUnit'" not in model:
        return False
    boundaries = _aedt_named_blocks(model, "Boundaries")
    setups = _aedt_named_blocks(model, "Setup_10GHz")
    solutions = _aedt_named_blocks(model, "SolutionManager")
    if len(boundaries) != 1 or len(setups) != 1 or len(solutions) != 1:
        return False
    for name, boundary_type in (
        ("PrimaryX", "Primary"),
        ("SecondaryX", "Secondary"),
        ("PrimaryY", "Primary"),
        ("SecondaryY", "Secondary"),
        ("FeedPort", "Lumped Port"),
        ("FloquetTop", "Floquet Port"),
    ):
        blocks = _aedt_named_blocks(boundaries[0], name)
        if len(blocks) != 1 or f"BoundType='{boundary_type}'" not in blocks[0]:
            return False
    if require_solved_adaptive:
        solution = solutions[0]
        if not (
            "Name='Setup_10GHz'" in solution
            and "Name='LastAdaptive'" in solution
            and "Soln(N='LastAdaptive'" in solution
        ):
            return False
    return True


def aedt_project_scan_state_matches(
    path: Path, state_key: tuple[float, float, float]
) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore")
    models = _aedt_named_blocks(text, "HFSSModel")
    if len(models) != 1:
        return False
    model = models[0]
    boundaries = _aedt_named_blocks(model, "Boundaries")
    setups = _aedt_named_blocks(model, "Setup_10GHz")
    solutions = _aedt_named_blocks(model, "SolutionManager")
    if len(boundaries) != 1 or len(setups) != 1 or len(solutions) != 1:
        return False
    scan_pairs = []
    for boundary_name in ("SecondaryX", "SecondaryY", "FloquetTop"):
        blocks = _aedt_named_blocks(boundaries[0], boundary_name)
        if len(blocks) != 1:
            return False
        match = re.search(
            r"PhaseDelay='UseScanAngle'\s+Phi='([+\-0-9.eE]+)deg'\s+"
            r"Theta='([+\-0-9.eE]+)deg'",
            blocks[0],
        )
        if not match:
            return False
        scan_pairs.append((float(match.group(1)), float(match.group(2))))
    expected_phi = state_key[2]
    expected_theta = state_key[1]
    if not all(
        math.isclose(float(phi), expected_phi, abs_tol=1.0e-9)
        and math.isclose(float(theta), expected_theta, abs_tol=1.0e-9)
        for phi, theta in scan_pairs[:3]
    ):
        return False
    setup_frequency = re.search(
        r"^\s*Frequency='([+\-0-9.eE]+)GHz'\s*$",
        setups[0],
        flags=re.MULTILINE,
    )
    if not setup_frequency or not math.isclose(
        float(setup_frequency.group(1)), state_key[0], abs_tol=5.0e-5
    ):
        return False
    floquet = _aedt_named_blocks(boundaries[0], "FloquetTop")[0]
    calculators = _aedt_named_blocks(floquet, "ModesCalculator")
    if len(calculators) != 1:
        return False
    calculator = calculators[0]
    required_properties = {
        "Frequency": state_key[0],
        "PhiStart": expected_phi,
        "PhiStop": expected_phi,
        "ThetaStart": expected_theta,
        "ThetaStop": expected_theta,
    }
    for property_name, expected_value in required_properties.items():
        match = re.search(
            rf"^\s*{property_name}='([+\-0-9.eE]+)(?:GHz|deg)'\s*$",
            calculator,
            flags=re.MULTILINE,
        )
        if not match or not math.isclose(
            float(match.group(1)), expected_value, abs_tol=5.0e-5
        ):
            return False
    solution = solutions[0]
    last_adaptive = re.search(
        r"\$begin 'Solution'\s+ID=\d+\s+Name='LastAdaptive'(?P<body>.*?)"
        r"\$end 'Solution'",
        solution,
        flags=re.DOTALL,
    )
    return bool(
        last_adaptive
        and re.search(
            rf"Column='{re.escape(f'{state_key[0]:g}')}GHz'",
            last_adaptive.group("body"),
        )
    )


def _periodic_state_metrics_from_artifacts(
    evidence: dict[str, Any],
    state_key: tuple[float, float, float],
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    model_state = json.loads(
        artifact_paths["model_state_audit"].read_text(encoding="utf-8")
    )
    model_key = (
        float(model_state["frequency_ghz"]),
        float(model_state["theta_deg"]),
        float(model_state["phi_deg"]),
    )
    if model_key != state_key:
        raise ValueError("Saved-model scan-state audit does not match row")
    source_names = [
        line.strip()
        for line in artifact_paths["source_names"].read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines()
        if line.strip()
    ]
    frequencies, matrices, _, port_names = parse_touchstone(
        artifact_paths["touchstone"], len(source_names)
    )
    if not port_names or set(port_names) != set(source_names):
        raise ValueError("Touchstone port mapping is incomplete")
    feed_indices = [
        index
        for index, name in enumerate(port_names)
        if name.split(":", 1)[0] == "FeedPort"
    ]
    floquet_indices = [
        index
        for index, name in enumerate(port_names)
        if name.split(":", 1)[0] == "FloquetTop"
    ]
    if len(feed_indices) != 1 or len(floquet_indices) != 2:
        raise ValueError("Expected one feed and two Floquet modes")
    frequency_index = int(np.argmin(np.abs(frequencies - state_key[0])))
    if abs(float(frequencies[frequency_index]) - state_key[0]) > 5.0e-5:
        raise ValueError("Touchstone does not contain the requested frequency")
    matrix = matrices[frequency_index]
    feed = feed_indices[0]
    gamma = complex(matrix[feed, feed])
    reflected = abs(gamma) ** 2
    accepted = 1.0 - reflected
    floquet = float(
        sum(abs(matrix[index, feed]) ** 2 for index in floquet_indices)
    )
    efficiency = floquet / accepted if accepted > 1.0e-12 else math.nan
    power_consistent = bool(
        math.isfinite(efficiency)
        and -1.0e-6 <= reflected <= 1.0 + 1.0e-6
        and -1.0e-6 <= floquet <= accepted + 1.0e-3
        and efficiency <= 1.001
    )
    with artifact_paths["gain_report"].open(
        encoding="utf-8-sig", errors="ignore"
    ) as handle:
        gain_rows = list(csv.DictReader(handle))
    matching_gain = [
        row
        for row in gain_rows
        if (
            abs(float(row["frequency_ghz"]) - state_key[0]) <= 5.0e-5
            and float(row["theta_deg"]) == state_key[1]
            and float(row["phi_deg"]) == state_key[2]
        )
    ]
    if len(matching_gain) != 1:
        raise ValueError("Gain report does not uniquely match scan state")
    convergence = _single_convergence_metrics(
        artifact_paths["convergence_profile"],
        artifact_paths["geometry_error"],
    )
    log_hits = critical_log_hits(
        artifact_paths["solver_log"].read_text(
            encoding="utf-8", errors="ignore"
        )
    )
    critical_count = sum(log_hits.values()) + int(
        convergence["small_segment_count"]
    )
    magnitude = abs(gamma)
    return {
        "active_rl_db": -20.0 * math.log10(max(magnitude, 1.0e-15)),
        "passive_rl_db": -20.0 * math.log10(max(magnitude, 1.0e-15)),
        "efficiency": efficiency,
        "realized_gain_db": float(matching_gain[0]["realized_gain_db"]),
        "final_delta_s": float(convergence["final_delta_s"]),
        "scan_blindness": bool(not power_consistent),
        "critical_warning_count": critical_count,
        "converged": bool(convergence["converged"]),
    }


def _verify_periodic_state_provenance(
    row: dict[str, Any],
    state_key: tuple[float, float, float],
    metric_fields: tuple[str, ...],
    ansys_executable_sha256: str,
) -> tuple[bool, dict[str, str]]:
    try:
        manifest_path = resolve(row["evidence_manifest_path"])
        expected_manifest_hash = row["evidence_manifest_sha256"]
        if (
            not manifest_path.is_file()
            or not _valid_sha256(expected_manifest_hash)
            or sha256(manifest_path) != expected_manifest_hash
        ):
            return False, {"verification_error": "manifest hash mismatch"}
        evidence = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_state = evidence["scan_state"]
        recorded_key = (
            float(recorded_state["frequency_ghz"]),
            float(recorded_state["theta_deg"]),
            float(recorded_state["phi_deg"]),
        )
        if recorded_key != state_key:
            return False, {"verification_error": "scan state mismatch"}
        if (
            evidence.get("case_id") != row.get("case_id")
            or evidence.get("evidence_source") != "HFSS_periodic_fullwave"
            or evidence.get("solve_complete") is not True
        ):
            return False, {"verification_error": "case metadata mismatch"}
        recorded_metrics = evidence["metrics"]
        for name in metric_fields:
            left = row.get(name)
            right = recorded_metrics.get(name)
            if isinstance(left, bool) or isinstance(right, bool):
                if left is not right:
                    return False, {
                        "verification_error": f"recorded {name} mismatch"
                    }
            elif float(left) != float(right):
                return False, {
                    "verification_error": f"recorded {name} mismatch"
                }
        artifact_paths: dict[str, Path] = {}
        for name in (
            "project",
            "touchstone",
            "source_names",
            "solver_script",
            "solver_log",
            "model_state_audit",
            "gain_report",
            "convergence_profile",
            "geometry_error",
            "run_audit",
        ):
            record = evidence["artifacts"][name]
            artifact = resolve(record["path"])
            expected_hash = record["sha256"]
            if (
                not artifact.is_file()
                or not _valid_sha256(expected_hash)
                or sha256(artifact) != expected_hash
            ):
                return False, {
                    "verification_error": f"{name} artifact mismatch"
                }
            artifact_paths[name] = artifact.resolve()
        if not validate_aedt_project_container(artifact_paths["project"]):
            return False, {
                "verification_error": "project is not an audited AEDT container"
            }
        if not aedt_project_scan_state_matches(
            artifact_paths["project"], state_key
        ):
            return False, {
                "verification_error": "AEDT project scan state mismatch"
            }
        solver_hash = evidence["artifacts"]["solver_script"]["sha256"]
        if (
            evidence.get("prepared_solver_sha256") != solver_hash
            or evidence.get("solver_sha256_executed") != solver_hash
        ):
            return False, {
                "verification_error": "prepared/executed solver hash mismatch"
            }
        run_audit = json.loads(
            artifact_paths["run_audit"].read_text(encoding="utf-8")
        )
        if not (
            run_audit.get("schema") == "v149_periodic_state_run_audit_v1"
            and run_audit.get("aedt_version") == "2023.1"
            and run_audit.get("ansys_executable_sha256")
            == ansys_executable_sha256
            and run_audit.get("started") is True
            and run_audit.get("blocked") is False
            and run_audit.get("return_code") == 0
            and run_audit.get("memory_aborted") is False
            and run_audit.get("solver_sha256_executed") == solver_hash
        ):
            return False, {
                "verification_error": "independent AEDT run audit failed"
            }
        audited_hashes = run_audit.get("artifact_sha256", {})
        for name, path in artifact_paths.items():
            if name == "run_audit":
                continue
            if audited_hashes.get(name) != sha256(path):
                return False, {
                    "verification_error": f"run audit did not bind {name}"
                }
        result_directory = resolve(run_audit["result_directory"])
        if result_directory.resolve() != manifest_path.parent.resolve():
            return False, {
                "verification_error": "run-audit result directory mismatch"
            }
        if not verify_convergence_artifacts(
            result_directory, run_audit.get("convergence_artifacts")
        ):
            return False, {
                "verification_error": "incomplete convergence artifact audit"
            }
        recomputed = _periodic_state_metrics_from_artifacts(
            evidence, state_key, artifact_paths
        )
        if recomputed["converged"] is not True:
            return False, {"verification_error": "solver did not converge"}
        for name in metric_fields:
            left = row.get(name)
            right = recomputed.get(name)
            if isinstance(left, bool) or isinstance(right, bool):
                if left is not right:
                    return False, {
                        "verification_error": f"recomputed {name} mismatch"
                    }
            elif not math.isclose(
                float(left), float(right), rel_tol=1.0e-7, abs_tol=1.0e-9
            ):
                return False, {
                    "verification_error": f"recomputed {name} mismatch"
                }
        return True, {
            **{
                name: str(path)
                for name, path in artifact_paths.items()
            },
            **{
                f"{name}_sha256": evidence["artifacts"][name]["sha256"]
                for name in artifact_paths
            },
            "evidence_manifest": str(manifest_path.resolve()),
            "evidence_manifest_sha256": expected_manifest_hash,
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        return False, {"verification_error": str(error)}


def aggregate_periodic_scan(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    validate_config(config)
    expected = {
        (
            float(state["frequency_ghz"]),
            float(state["theta_deg"]),
            float(state["phi_deg"]),
        )
        for state in scan_states(config)
    }
    keyed: dict[tuple[float, float, float], dict[str, Any]] = {}
    duplicate_state = False
    for row in rows:
        key = (
            float(row["frequency_ghz"]),
            float(row["theta_deg"]),
            float(row["phi_deg"]),
        )
        if key in keyed:
            duplicate_state = True
        keyed[key] = row
    complete_rows = [keyed[key] for key in sorted(expected) if key in keyed]
    numeric_fields = (
        "active_rl_db",
        "passive_rl_db",
        "efficiency",
        "realized_gain_db",
        "final_delta_s",
    )
    finite_rows = all(
        all(
            isinstance(row.get(name), (int, float))
            and math.isfinite(float(row[name]))
            for name in numeric_fields
        )
        for row in complete_rows
    )
    metric_fields = (
        *numeric_fields,
        "scan_blindness",
        "critical_warning_count",
    )
    ansys_executable = resolve(config["ansys_executable"])
    ansys_executable_hash = (
        sha256(ansys_executable) if ansys_executable.is_file() else ""
    )
    provenance_results = [
        _verify_periodic_state_provenance(
            row, key, metric_fields, ansys_executable_hash
        )
        for key, row in zip(sorted(expected), complete_rows)
    ]
    provenance_complete = bool(
        complete_rows
        and finite_rows
        and not duplicate_state
        and len(rows) == len(expected)
        and all(
            row.get("provenance_verified") is True and verified
            for row, (verified, _) in zip(
                complete_rows, provenance_results
            )
        )
    )
    artifact_paths = [paths for _, paths in provenance_results]
    if provenance_complete:
        for name in (
            "project",
            "touchstone",
            "source_names",
            "solver_script",
            "solver_log",
            "model_state_audit",
            "gain_report",
            "convergence_profile",
            "geometry_error",
            "run_audit",
            "evidence_manifest",
        ):
            values = [paths[name] for paths in artifact_paths]
            if len(set(values)) != len(expected):
                provenance_complete = False
                break
        for name in (
            "project_sha256",
            "model_state_audit_sha256",
            "evidence_manifest_sha256",
        ):
            values = [paths[name] for paths in artifact_paths]
            if len(set(values)) != len(expected):
                provenance_complete = False
                break
    # v1.49 stage A intentionally has no AEDT-reopen batch attestor yet.
    # Keep aggregation diagnostic-only until a separately versioned workflow
    # opens every solved project in AEDT and verifies its saved solution.
    physical_attestation_complete = False
    evidence_complete = bool(
        set(keyed) == expected
        and provenance_complete
        and physical_attestation_complete
        and all(
            row.get("evidence_source") == "HFSS_periodic_fullwave"
            and row.get("solve_complete") is True
            for row in complete_rows
        )
    )
    if not complete_rows:
        return {
            "evidence_source": "HFSS_periodic_fullwave",
            "evidence_complete": False,
            "provenance_complete": False,
            "physical_attestation_complete": False,
            "physical_gate_armed": False,
            "scan_state_count": 0,
            "minimum_active_rl_db": -math.inf,
            "broadside_passive_rl_db": -math.inf,
            "minimum_efficiency": -math.inf,
            "maximum_scan_gain_drop_db": math.inf,
            "maximum_final_delta_s": math.inf,
            "scan_blindness_detected": True,
            "critical_warning_count": 10**9,
        }
    if not finite_rows:
        return {
            "evidence_source": "HFSS_periodic_fullwave",
            "evidence_complete": False,
            "provenance_complete": False,
            "physical_attestation_complete": False,
            "physical_gate_armed": False,
            "scan_state_count": len(complete_rows),
            "minimum_active_rl_db": -math.inf,
            "broadside_passive_rl_db": -math.inf,
            "minimum_efficiency": -math.inf,
            "maximum_scan_gain_drop_db": math.inf,
            "maximum_final_delta_s": math.inf,
            "scan_blindness_detected": True,
            "critical_warning_count": 10**9,
        }
    broadside = [
        row for row in complete_rows if float(row["theta_deg"]) == 0.0
    ]
    gain_drop = 0.0
    for row in complete_rows:
        references = [
            candidate
            for candidate in broadside
            if float(candidate["frequency_ghz"])
            == float(row["frequency_ghz"])
            and float(candidate["phi_deg"]) == float(row["phi_deg"])
        ]
        if references:
            gain_drop = max(
                gain_drop,
                float(references[0]["realized_gain_db"])
                - float(row["realized_gain_db"]),
            )
    return {
        "evidence_source": "HFSS_periodic_fullwave",
        "evidence_complete": evidence_complete,
        "provenance_complete": provenance_complete,
        "physical_attestation_complete": physical_attestation_complete,
        "physical_gate_armed": False,
        "physical_gate_lock_reason": (
            "v1.49 stage A lacks an independent AEDT reopen attestor; "
            "aggregated metrics are diagnostic only."
        ),
        "scan_state_count": len(complete_rows),
        "minimum_active_rl_db": min(
            float(row["active_rl_db"]) for row in complete_rows
        ),
        "broadside_passive_rl_db": min(
            float(row["passive_rl_db"]) for row in broadside
        )
        if broadside
        else -math.inf,
        "minimum_efficiency": min(
            float(row["efficiency"]) for row in complete_rows
        ),
        "maximum_scan_gain_drop_db": gain_drop,
        "maximum_final_delta_s": max(
            float(row["final_delta_s"]) for row in complete_rows
        ),
        "scan_blindness_detected": any(
            bool(row.get("scan_blindness")) for row in complete_rows
        ),
        "critical_warning_count": sum(
            int(row.get("critical_warning_count", 0))
            for row in complete_rows
        ),
    }


def resolve_build_gate_path(run_root: Path) -> Path:
    run_root = run_root.resolve()
    local_gate = run_root / "periodic" / "build_smoke" / "build_gate.json"
    if local_gate.is_file():
        return local_gate.resolve()

    continuation_path = run_root / "continuation_audit.json"
    authorization_path = run_root / "solve_authorization.json"
    if not continuation_path.is_file() or not authorization_path.is_file():
        raise FileNotFoundError(local_gate)
    continuation = json.loads(continuation_path.read_text(encoding="utf-8"))
    authorization = json.loads(
        authorization_path.read_text(encoding="utf-8")
    )
    source_run = Path(continuation.get("source_run", "")).resolve()
    source_evidence = authorization.get("source_evidence", {})
    authorized_source = Path(source_evidence.get("source_run", "")).resolve()
    if source_run != authorized_source:
        raise RuntimeError("Continuation source run does not match authorization")
    source_gate = (
        source_run / "periodic" / "build_smoke" / "build_gate.json"
    ).resolve()
    if not source_gate.is_file():
        raise FileNotFoundError(source_gate)
    expected_hash = source_evidence.get("source_build_gate_sha256")
    if not _valid_sha256(expected_hash) or sha256(source_gate) != expected_hash:
        raise RuntimeError("Continuation source build gate hash mismatch")
    return source_gate


def finalize_stage_summary(
    run_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    for immutable in (
        run_root / "stage_summary.json",
        run_root / "stage_summary.md",
        run_root / "sha256_manifest.csv",
    ):
        if immutable.exists():
            raise FileExistsError(
                f"Refusing to overwrite finalized evidence: {immutable}"
            )
    build_gate_path = resolve_build_gate_path(run_root)
    build_gate = json.loads(build_gate_path.read_text(encoding="utf-8"))
    solve_folder = run_root / "periodic" / "nominal_solve"
    nominal_analysis_path = solve_folder / "nominal_analysis.json"
    nominal_analysis = (
        json.loads(nominal_analysis_path.read_text(encoding="utf-8"))
        if nominal_analysis_path.is_file()
        else None
    )
    nominal_rows = nominal_analysis.get("rows", []) if nominal_analysis else []
    nominal_diagnostic_passed = bool(
        nominal_analysis
        and nominal_analysis.get("nominal_export_evidence_complete") is True
        and len(nominal_rows) == len(config["frequencies_ghz"])
        and all(
            float(row["active_rl_db"])
            >= float(config["gates"]["minimum_periodic_active_rl_db"])
            and float(row["passive_rl_db"])
            >= float(config["gates"]["minimum_broadside_passive_rl_db"])
            and float(row["accepted_power_efficiency"])
            >= float(config["gates"]["minimum_periodic_efficiency"])
            for row in nominal_rows
        )
    )
    blocked_files = sorted(solve_folder.glob("preflight_block_*.json"))
    latest_block = (
        json.loads(blocked_files[-1].read_text(encoding="utf-8"))
        if blocked_files
        else None
    )
    current_status = status(config)
    summary = {
        "protocol": config["protocol"],
        "run_root": str(run_root.resolve()),
        "stage": (
            "periodic_nominal_solve_analyzed"
            if nominal_analysis
            else "periodic_build_complete_nominal_solve_pending"
        ),
        "periodic_build_smoke_passed": bool(
            build_gate.get("build_smoke_passed")
        ),
        "build_gate_path": str(build_gate_path),
        "build_gate_sha256": sha256(build_gate_path),
        "build_evidence_source": build_gate.get("evidence_source"),
        "physical_hfss_metrics_available": bool(nominal_rows),
        "periodic_physical_gate_evaluated": False,
        "nominal_diagnostic_passed": nominal_diagnostic_passed,
        "nominal_analysis_path": (
            str(nominal_analysis_path.resolve()) if nominal_analysis else None
        ),
        "nominal_analysis": nominal_analysis,
        "periodic_doe_sample_count": len(generate_periodic_doe(config)),
        "periodic_doe_hfss_batch_authorized": False,
        "nominal_solve_prepared": (
            solve_folder / "case_manifest.json"
        ).exists(),
        "nominal_solve_started": (
            solve_folder / "run_audit.json"
        ).exists(),
        "latest_solve_preflight_block": latest_block,
        "current_resource_status": current_status,
        "locked_stages": {
            "periodic_doe_batch": True,
            "finite_1x1": True,
            "finite_2x2": True,
            "finite_4x4": True,
            "array_16x16": True,
            "eep_export": True,
            "training_labels": True,
            "critic_retraining": True,
        },
        "decision": (
            "STOP_AFTER_NOMINAL_DIAGNOSTIC_FAILURE"
            if nominal_analysis and not nominal_diagnostic_passed
            else (
                "REVIEW_NOMINAL_BEFORE_PERIODIC_DOE"
                if nominal_diagnostic_passed
                else (
                    "CONTINUE_LATER_WITH_ONE_NOMINAL_PERIODIC_SOLVE"
                    if build_gate.get("build_smoke_passed")
                    else "STOP_AND_REPAIR_PERIODIC_CAD"
                )
            )
        ),
        "next_permitted_action": (
            "Repair the periodic feed/input geometry in a new immutable run; "
            "do not start DOE or downstream array stages."
            if nominal_analysis and not nominal_diagnostic_passed
            else (
                "Review the nominal diagnostic evidence before explicitly "
                "authorizing any periodic DOE batch."
                if nominal_diagnostic_passed
                else (
                "When available RAM is at least 13 GiB and no AEDT process "
                "exists, run only the frozen broadside nominal periodic "
                "9.8-10.2 GHz solve. Do not start DOE, 1x1, 2x2, 4x4, "
                "16x16, labels, or critic."
                if build_gate.get("build_smoke_passed")
                else (
                    "Repair and repeat the periodic native-CAD build smoke "
                    "in a new immutable run. No physical solve is authorized."
                )
                )
            )
        ),
    }
    write_json(run_root / "stage_summary.json", summary)
    lines = [
        "# v1.49 Stage Summary",
        "",
        "## Measured Evidence",
        "",
        f"- Periodic native CAD build gate: {'PASS' if summary['periodic_build_smoke_passed'] else 'FAIL'}.",
        (
            "- Evidence scope: completed HFSS nominal broadside diagnostic; "
            "the 45-state periodic physical gate is not evaluated."
            if nominal_analysis
            else "- Evidence scope: HFSS build/import only; no mesh solution or antenna metric is claimed."
        ),
        f"- DOE manifest: {summary['periodic_doe_sample_count']} frozen geometry candidates; HFSS batch remains unauthorized.",
        f"- Current free memory: {float(current_status['free_memory_gib']):.2f} GiB.",
        f"- Nominal solve preflight: {'PASS' if current_status['solve_preflight_pass'] else 'BLOCKED'} (requires 13 GiB and zero AEDT instances).",
        "",
        "## Gate Decision",
        "",
        "- Periodic physical gate has not been evaluated.",
        f"- Nominal diagnostic gate: {'PASS' if nominal_diagnostic_passed else 'FAIL' if nominal_analysis else 'NOT RUN'}.",
        "- Finite 1x1, 2x2, 4x4, 16x16, EEP export, training labels, and critic retraining remain locked.",
        (
            "- The completed run is blocked by a nominal antenna/input diagnostic failure."
            if nominal_analysis and not nominal_diagnostic_passed
            else "- No nominal antenna-performance failure has been established."
        ),
        "",
        "## Next Permitted Action",
        "",
        summary["next_permitted_action"],
    ]
    (run_root / "stage_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )
    manifest_path = run_root / "sha256_manifest.csv"
    rows: list[dict[str, Any]] = []
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(run_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv(manifest_path, rows)
    summary["sha256_manifest_path"] = str(manifest_path.resolve())
    summary["hashed_file_count"] = len(rows)
    return summary


def status(config: dict[str, Any]) -> dict[str, Any]:
    processes = aedt_processes()
    free = memory_available_gib()
    return {
        "protocol": config["protocol"],
        "free_memory_gib": free,
        "aedt_process_count": len(processes),
        "aedt_processes": processes,
        "solve_preflight_pass": memory_allows_stage(
            config, free, len(processes)
        ),
        "locked_stages": [
            name
            for name, allowed in config["scope"].items()
            if name.startswith("allow_") and not allowed
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "validate-config",
            "status",
            "preregister",
            "create-continuation",
            "prepare-build-smoke",
            "run-build-smoke",
            "audit-build-smoke",
            "prepare-doe-manifest",
            "prepare-nominal-solve",
            "run-nominal-solve",
            "analyze-nominal-solve",
            "finalize-stage-summary",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--source-run-root", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    if args.command == "validate-config":
        result = {"valid": True}
    elif args.command == "status":
        result = status(config)
    elif args.command == "preregister":
        result = preregister(config)
    elif args.command == "create-continuation":
        if args.source_run_root is None:
            parser.error("create-continuation requires --source-run-root")
        result = create_nominal_continuation(resolve(args.source_run_root))
    else:
        if args.run_root is None:
            parser.error(f"{args.command} requires --run-root")
        run_root = resolve(args.run_root)
        config = load_run_config(run_root)
        if args.command == "prepare-build-smoke":
            result = prepare_periodic_build_smoke(run_root, config)
        elif args.command == "run-build-smoke":
            result = run_periodic_build_smoke(run_root, config)
        elif args.command == "prepare-doe-manifest":
            result = prepare_doe_manifest(run_root, config)
        elif args.command == "prepare-nominal-solve":
            result = prepare_nominal_periodic_solve(run_root, config)
        elif args.command == "run-nominal-solve":
            result = run_nominal_periodic_solve(run_root, config)
        elif args.command == "analyze-nominal-solve":
            result = analyze_nominal_periodic_solve(run_root, config)
        elif args.command == "finalize-stage-summary":
            result = finalize_stage_summary(run_root, config)
        else:
            result = audit_periodic_build_smoke(run_root, config)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
