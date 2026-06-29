import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOMFORMER_ROOT = Path(__file__).resolve().parents[2] / "baselines" / "RoomFormer"
sys.path.insert(0, str(ROOMFORMER_ROOT / "s3d_floorplan_eval" / "S3DLoader"))

from s3d_utils import generate_floorplan, parse_floor_plan_polys  # noqa: E402


TYPE_DOOR = 16
TYPE_WINDOW = 17


def canonical_edge(p1, p2):
    a = tuple(np.round(p1, 4))
    b = tuple(np.round(p2, 4))
    return (a, b) if a <= b else (b, a)


def collect_edges(polygons):
    seen = set()
    edges = []
    for poly in polygons:
        points = np.asarray(poly, dtype=float)
        if len(points) < 2:
            continue

        for idx in range(len(points)):
            p1 = points[idx]
            p2 = points[(idx + 1) % len(points)]
            if np.linalg.norm(p2 - p1) < 1e-6:
                continue

            key = canonical_edge(p1, p2)
            if key in seen:
                continue

            seen.add(key)
            edges.append((p1, p2))
    return edges


def line_distance_to_segment_midpoint(line, segment):
    line = np.asarray(line, dtype=float)
    p1, p2 = np.asarray(segment[0], dtype=float), np.asarray(segment[1], dtype=float)
    line_vec = line[1] - line[0]
    seg_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    seg_len = np.linalg.norm(seg_vec)
    if line_len < 1e-6 or seg_len < 1e-6:
        return float("inf")

    line_dir = line_vec / line_len
    seg_dir = seg_vec / seg_len
    parallel_penalty = abs(np.cross(line_dir, seg_dir))
    line_mid = line.mean(axis=0)
    seg_mid = (p1 + p2) * 0.5

    # Distance to the infinite wall line plus a small midpoint preference.
    normal_dist = abs(np.cross(seg_dir, line_mid - p1))
    mid_dist = np.linalg.norm(line_mid - seg_mid) / max(seg_len, 1e-6)
    return normal_dist + 5.0 * parallel_penalty + 0.1 * mid_dist


def nearest_wall_id(line, edges):
    if not edges:
        return None
    scores = [line_distance_to_segment_midpoint(line, edge) for edge in edges]
    return int(np.argmin(scores))


def scale_point(point, scale):
    point = np.asarray(point, dtype=float) * scale
    return point


def layout_to_spatiallm(edges, openings, output_path, scale):
    lines = []
    for wall_id, (p1, p2) in enumerate(edges):
        a = scale_point(p1, scale)
        b = scale_point(p2, scale)
        lines.append(
            "wall_{id}=Wall({ax:.6f},{ay:.6f},0,{bx:.6f},{by:.6f},0,3.0,0.05)".format(
                id=wall_id, ax=a[0], ay=a[1], bx=b[0], by=b[1]
            )
        )

    door_id = 0
    window_id = 0
    for opening_line, opening_type in openings:
        wall_id = nearest_wall_id(opening_line, edges)
        if wall_id is None:
            continue

        points = np.asarray(opening_line, dtype=float)
        center = scale_point(points.mean(axis=0), scale)
        width = np.linalg.norm((points[1] - points[0]) * scale)
        if opening_type == TYPE_DOOR:
            lines.append(
                "door_{id}=Door(wall_{wall_id},{x:.6f},{y:.6f},1.0,{width:.6f},2.0)".format(
                    id=door_id, wall_id=wall_id, x=center[0], y=center[1], width=width
                )
            )
            door_id += 1
        elif opening_type == TYPE_WINDOW:
            lines.append(
                "window_{id}=Window(wall_{wall_id},{x:.6f},{y:.6f},1.5,{width:.6f},1.2)".format(
                    id=window_id,
                    wall_id=wall_id,
                    x=center[0],
                    y=center[1],
                    width=width,
                )
            )
            window_id += 1

    output_path.write_text("\n".join(lines))


def load_gt(scene_id, montefloor_data_dir):
    scene_path = montefloor_data_dir / "test" / f"scene_{scene_id:05d}" / "annotation_3d.json"
    annos = json.loads(scene_path.read_text())
    polygons = parse_floor_plan_polys(annos)
    _, room_polys, _ = generate_floorplan(
        annos,
        polygons,
        height=256,
        width=256,
        ignore_types=["outwall", "door", "window"],
        constant_color=False,
        shuffle=False,
    )
    _, opening_lines, opening_types = generate_floorplan(
        annos,
        polygons,
        height=256,
        width=256,
        ignore_types=[],
        include_types=["door", "window"],
        fillpoly=False,
        constant_color=True,
        shuffle=False,
    )
    return room_polys, list(zip(opening_lines, opening_types))


def load_prediction(prediction_path):
    data = json.loads(prediction_path.read_text())
    openings = list(zip(data["window_doors"], data["window_doors_types"]))
    return data["room_polys"], openings


def write_metadata(scene_ids, output_dir):
    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "pcd", "layout"])
        writer.writeheader()
        for scene_id in scene_ids:
            writer.writerow({"id": str(scene_id), "pcd": "", "layout": f"{scene_id}.txt"})

    label_mapping_path = output_dir / "label_mapping.tsv"
    label_mapping_path.write_text("spatiallm59\tspatiallm20\n")


def convert(args):
    prediction_dir = Path(args.prediction_dir)
    montefloor_data_dir = Path(args.montefloor_data_dir)
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "pred"
    gt_dir = output_dir / "gt"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    scale = args.world_size / args.image_size
    scene_ids = []
    for prediction_path in sorted(prediction_dir.glob("*.json")):
        scene_id = int(prediction_path.stem)
        scene_ids.append(scene_id)

        pred_polys, pred_openings = load_prediction(prediction_path)
        pred_edges = collect_edges(pred_polys)
        layout_to_spatiallm(pred_edges, pred_openings, pred_dir / f"{scene_id}.txt", scale)

        gt_polys, gt_openings = load_gt(scene_id, montefloor_data_dir)
        gt_edges = collect_edges(gt_polys)
        layout_to_spatiallm(gt_edges, gt_openings, gt_dir / f"{scene_id}.txt", scale)

    write_metadata(scene_ids, output_dir)
    print(f"Converted {len(scene_ids)} scenes to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_dir",
        default="baselines/RoomFormer/checkpoints/eval_stru3d_sem_rich/predictions",
    )
    parser.add_argument(
        "--montefloor_data_dir",
        default="baselines/RoomFormer/s3d_floorplan_eval/montefloor_data",
    )
    parser.add_argument("--output_dir", default="baselines/RoomFormer/spatiallm_eval")
    parser.add_argument("--image_size", type=float, default=256.0)
    parser.add_argument("--world_size", type=float, default=32.0)
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
