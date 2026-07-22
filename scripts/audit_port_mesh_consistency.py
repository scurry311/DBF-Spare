#!/usr/bin/env python3
"""Audit the 256 HFSS lumped ports and deterministic feed-mesh coverage."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


NPORTS = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--g3derr", type=Path, required=True)
    parser.add_argument("--stability-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def number(value: str) -> float:
    match = re.search(r"[+\-0-9.eE]+", value)
    if match is None:
        raise ValueError(value)
    return float(match.group())


def named_blocks(text: str, kind: str, name_pattern: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        rf"\$begin '({name_pattern})'(.*?)\$end '\1'", re.DOTALL
    )
    return [(match.group(1), match.group(2)) for match in pattern.finditer(text)]


def geometry_parts(text: str) -> dict[str, str]:
    parts: dict[str, str] = {}
    for match in re.finditer(
        r"\$begin 'GeometryPart'(.*?)\$end 'GeometryPart'", text, re.DOTALL
    ):
        block = match.group(1)
        name = re.search(r"\bName='([^']+)'", block)
        if name:
            parts[name.group(1)] = block
    return parts


def rectangle(block: str) -> dict[str, object]:
    params = re.search(
        r"\$begin 'RectangleParameters'(.*?)\$end 'RectangleParameters'",
        block,
        re.DOTALL,
    )
    if params is None:
        raise ValueError("RectangleParameters missing")
    body_ids = [int(value) for value in re.findall(r"\bBodyID=(\d+)", block)]
    face_ids = [int(value) for value in re.findall(r"\bStartFaceID=(\d+)", block)]
    values = params.group(1)
    result: dict[str, object] = {
        "body_id": body_ids[0],
        "face_id": face_ids[-1],
    }
    for field in ("XStart", "YStart", "ZStart", "Width", "Height", "WhichAxis"):
        match = re.search(rf"\b{field}='([^']+)'", values)
        if match:
            result[field] = match.group(1) if field == "WhichAxis" else number(match.group(1))
    return result


def box(block: str) -> dict[str, float]:
    params = re.search(r"\$begin 'BoxParameters'(.*?)\$end 'BoxParameters'", block, re.DOTALL)
    if params is None:
        raise ValueError("BoxParameters missing")
    result = {}
    for field in ("XPosition", "YPosition", "ZPosition", "XSize", "YSize", "ZSize"):
        match = re.search(rf"\b{field}='([^']+)'", params.group(1))
        if match:
            result[field] = number(match.group(1))
    return result


def parse_ports(text: str, parts: dict[str, str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, block in named_blocks(text, "boundary", r"P\d{3}"):
        index = int(name[1:])
        positions = re.findall(
            r"\$begin 'GeometryPosition'(.*?)\$end 'GeometryPosition'", block, re.DOTALL
        )
        coords: list[dict[str, object]] = []
        for position in positions:
            item: dict[str, object] = {}
            for field in ("EntityID", "UParam", "VParam", "XPosition", "YPosition", "ZPosition"):
                match = re.search(rf"\b{field}=(?:'([^']+)'|([^\r\n]+))", position)
                if match:
                    raw = (match.group(1) or match.group(2)).strip()
                    item[field] = int(raw) if field == "EntityID" else number(raw)
            item["attached"] = "IsAttachedToEntity=true" in position
            item["on_face"] = "PositionType='OnFace'" in position
            coords.append(item)
        port_sheet = rectangle(parts[f"PortSheet_{index:03d}"])
        assignment = re.search(r"\bObjects\((\d+)\)", block)
        row, column = divmod(index, 16)
        rows.append(
            {
                "port_1based": index + 1,
                "row_0based": row,
                "column_0based": column,
                "assignment_body_id": int(assignment.group(1)) if assignment else -1,
                "port_sheet_body_id": port_sheet["body_id"],
                "port_sheet_face_id": port_sheet["face_id"],
                "port_sheet_width_mm": port_sheet["Width"],
                "port_sheet_height_mm": port_sheet["Height"],
                "port_sheet_z_start_mm": port_sheet["ZStart"],
                "modal": "LumpedPortType='Modal'" in block,
                "mode_count": len(re.findall(r"\$begin 'Mode\d+'", block)),
                "mode_num": int(re.search(r"\bModeNum=(\d+)", block).group(1)),
                "use_int_line": "UseIntLine=true" in block,
                "char_imp_zpi": "CharImp='Zpi'" in block,
                "impedance_50ohm": "Impedance='50ohm'" in block,
                "do_deembed_false": "DoDeembed=false" in block,
                "line_position_count": len(coords),
                "line_start_entity_id": coords[0].get("EntityID", -1),
                "line_end_entity_id": coords[1].get("EntityID", -1),
                "line_start_u": coords[0].get("UParam", np.nan),
                "line_end_u": coords[1].get("UParam", np.nan),
                "line_start_v": coords[0].get("VParam", np.nan),
                "line_end_v": coords[1].get("VParam", np.nan),
                "line_same_xy": bool(
                    coords[0].get("XPosition") == coords[1].get("XPosition")
                    and coords[0].get("YPosition") == coords[1].get("YPosition")
                ),
                "line_attached_on_face": bool(
                    len(coords) == 2
                    and all(item.get("attached") and item.get("on_face") for item in coords)
                ),
            }
        )
    rows.sort(key=lambda item: int(item["port_1based"]))
    return rows


def parse_mesh_operation(text: str, name: str) -> dict[str, object]:
    match = re.search(rf"\$begin '{re.escape(name)}'(.*?)\$end '{re.escape(name)}'", text, re.DOTALL)
    if match is None:
        return {"present": False, "name": name}
    block = match.group(1)
    objects = re.search(r"Objects\(([^)]*)\)", block, re.DOTALL)
    ids = [int(value) for value in re.findall(r"\d+", objects.group(1))] if objects else []
    length = re.search(r"MaxLength='([^']+)'", block)
    return {
        "present": True,
        "name": name,
        "enabled": "Enabled=true" in block,
        "refine_inside": "RefineInside=true" in block,
        "object_count": len(ids),
        "object_ids": ids,
        "max_length_mm": number(length.group(1)) if length else None,
    }


def small_segments(path: Path) -> tuple[list[dict[str, object]], Counter[str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for block in re.findall(r"BEGIN_ERR(.*?)END_ERR", text, re.DOTALL):
        body = re.search(r"Small mesh segment detected on body\s*:\s*([^\r\n]+)", block, re.I)
        length = re.search(r"Segment length\(s\)\s*:\s*([0-9.eE+\-]+)mm", block, re.I)
        if body and length:
            body_name = body.group(1).strip()
            counts[body_name] += 1
            rows.append({"body": body_name, "length_mm": float(length.group(1))})
    return rows, counts


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    text = args.project.read_text(encoding="utf-8", errors="ignore")
    parts = geometry_parts(text)
    ports = parse_ports(text, parts)
    if len(ports) != NPORTS:
        raise ValueError(f"Parsed {len(ports)} ports")
    errors, body_counts = small_segments(args.g3derr)
    stability = {
        int(row["port_1based"]): row
        for row in csv.DictReader(args.stability_csv.open(encoding="utf-8-sig"))
    }
    for row in ports:
        port = int(row["port_1based"])
        row["diagonal_delta_s"] = float(stability[port]["diagonal_delta_s"])
        row["patch_small_segment_count"] = body_counts.get(f"Patch_{port - 1:03d}", 0)
        row["assignment_matches_sheet"] = row["assignment_body_id"] == row["port_sheet_body_id"]
        row["line_entity_matches_sheet_face"] = bool(
            row["line_start_entity_id"] == row["line_end_entity_id"] == row["port_sheet_face_id"]
        )
        row["line_u_direction_consistent"] = float(row["line_start_u"]) > float(row["line_end_u"])
        row["line_endpoint_offset_mm"] = float(row["port_sheet_height_mm"]) * (
            1.0 - float(row["line_start_u"])
        )
    mesh_ops = {
        name: parse_mesh_operation(text, name)
        for name in (
            "PortFeedUniform_0p180mm",
            "FeedSheetUniform_0p180mm",
            "FeedNeighborhoodUniform_0p180mm",
        )
    }
    port_body_ids = {int(row["port_sheet_body_id"]) for row in ports}
    feed_seed_ids = {rectangle(parts[f"FeedSeed_{index:03d}"])["body_id"] for index in range(NPORTS)}
    port_op_ids = set(mesh_ops["PortFeedUniform_0p180mm"].get("object_ids", []))
    seed_op_ids = set(mesh_ops["FeedSheetUniform_0p180mm"].get("object_ids", []))
    geometry = {
        "substrate": box(parts["Substrate"]),
        "ground": box(parts["Ground"]),
        "patch_000": box(parts["Patch_000"]),
        "port_sheet_000": rectangle(parts["PortSheet_000"]),
        "feed_seed_000": rectangle(parts["FeedSeed_000"]),
    }
    delta = np.asarray([float(row["diagonal_delta_s"]) for row in ports])
    segment_count = np.asarray([int(row["patch_small_segment_count"]) for row in ports])
    correlation = float(np.corrcoef(delta, np.log1p(segment_count))[0, 1])
    consistent_fields = (
        "assignment_matches_sheet",
        "line_entity_matches_sheet_face",
        "line_same_xy",
        "line_attached_on_face",
        "line_u_direction_consistent",
        "modal",
        "use_int_line",
        "char_imp_zpi",
        "impedance_50ohm",
        "do_deembed_false",
    )
    summary = {
        "project": str(args.project.resolve()),
        "port_count": len(ports),
        "port_consistency_failures": {
            field: sum(not bool(row[field]) for row in ports) for field in consistent_fields
        },
        "mode_count_values": sorted({int(row["mode_count"]) for row in ports}),
        "mode_num_values": sorted({int(row["mode_num"]) for row in ports}),
        "port_sheet_width_mm_values": sorted({float(row["port_sheet_width_mm"]) for row in ports}),
        "port_sheet_height_mm_values": sorted({float(row["port_sheet_height_mm"]) for row in ports}),
        "line_endpoint_offset_mm_range": [
            min(float(row["line_endpoint_offset_mm"]) for row in ports),
            max(float(row["line_endpoint_offset_mm"]) for row in ports),
        ],
        "geometry": geometry,
        "mesh_operations": mesh_ops,
        "port_mesh_exact_coverage": port_op_ids == port_body_ids,
        "feed_seed_mesh_exact_coverage": seed_op_ids == feed_seed_ids,
        "volumetric_feed_neighborhood_present": mesh_ops[
            "FeedNeighborhoodUniform_0p180mm"
        ]["present"],
        "small_segment_error_block_count": len(errors),
        "small_segment_min_length_mm": min(float(row["length_mm"]) for row in errors),
        "small_segment_max_length_mm": max(float(row["length_mm"]) for row in errors),
        "small_segment_top_bodies": body_counts.most_common(30),
        "patch_small_segment_vs_diagonal_delta_log_pearson": correlation,
        "decision": "block_pass6_fix_mesh_coverage_and_small_segments",
        "training_labels_locked": True,
    }
    write_csv(args.out_dir / "port_consistency_audit.csv", ports)
    write_csv(args.out_dir / "small_segment_records.csv", errors)
    write_csv(
        args.out_dir / "small_segment_body_counts.csv",
        [{"body": name, "count": count} for name, count in body_counts.most_common()],
    )
    (args.out_dir / "port_mesh_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
