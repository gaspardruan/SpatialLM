import json
from pathlib import Path

import numpy as np

from spatiallm.pcd import get_points_and_colors, load_o3d_pcd


DEFAULT_DATASET_DIR = (
    "/ssd/zq/.cache/huggingface/hub/datasets--ysmao--structured3d-spatiallm/"
    "snapshots/c5bedd45675b566547e6ae0bc077681bc58b7b35"
)

TYPE_DOOR = 16
TYPE_WINDOW = 17


def scene_stem(scene_id):
    scene_id = str(scene_id)
    if scene_id.startswith("scene_"):
        return scene_id
    return f"scene_{int(scene_id):05d}"


def scene_number(scene_id):
    return int(scene_stem(scene_id).split("_")[-1])


def roomformer_normalization(points):
    # Mirrors RoomFormer data_preprocess/stru3d/stru3d_utils.py::generate_density.
    ps = points.copy() * -1
    ps[:, 0] *= -1
    ps[:, 1] *= -1

    max_coords = np.max(ps, axis=0)
    min_coords = np.min(ps, axis=0)
    extent = max_coords - min_coords
    return {
        "min_coords": min_coords - 0.1 * extent,
        "max_coords": max_coords + 0.1 * extent,
        "image_res": np.array([256.0, 256.0]),
    }


def image_to_world_xy(point_xy, norm):
    point_xy = np.asarray(point_xy, dtype=float)
    min_xy = norm["min_coords"][:2]
    max_xy = norm["max_coords"][:2]
    image_res = norm["image_res"]
    return point_xy / image_res * (max_xy - min_xy) + min_xy


def canonical_edge(p1, p2):
    a = tuple(np.round(p1, 4))
    b = tuple(np.round(p2, 4))
    return (a, b) if a <= b else (b, a)


def collect_edges(room_polys, norm):
    seen = set()
    edges = []
    for poly in room_polys:
        points = np.asarray([image_to_world_xy(p, norm) for p in poly], dtype=float)
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


def line_distance_to_segment(line, segment):
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
    normal_dist = abs(np.cross(seg_dir, line_mid - p1))
    mid_dist = np.linalg.norm(line_mid - seg_mid) / max(seg_len, 1e-6)
    return normal_dist + 5.0 * parallel_penalty + 0.1 * mid_dist


def nearest_wall_id(line, edges):
    if not edges:
        return None
    return int(np.argmin([line_distance_to_segment(line, edge) for edge in edges]))


def opening_wall_candidates(
    line,
    edges,
    max_normal_distance=0.25,
    min_overlap_ratio=0.5,
    max_candidates=2,
):
    line = np.asarray(line, dtype=float)
    line_vec = line[1] - line[0]
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6:
        return []

    candidates = []
    for wall_id, (p1, p2) in enumerate(edges):
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        wall_vec = p2 - p1
        wall_len = np.linalg.norm(wall_vec)
        if wall_len < 1e-6:
            continue

        wall_dir = wall_vec / wall_len
        line_dir = line_vec / line_len
        if abs(np.cross(wall_dir, line_dir)) > 0.1:
            continue

        line_proj = np.sort([(point - p1).dot(wall_dir) for point in line])
        wall_proj = np.array([0.0, wall_len])
        overlap = min(line_proj[1], wall_proj[1]) - max(line_proj[0], wall_proj[0])
        if overlap / line_len < min_overlap_ratio:
            continue

        normal_dist = abs(np.cross(wall_dir, line.mean(axis=0) - p1))
        if normal_dist > max_normal_distance:
            continue

        candidates.append((normal_dist, wall_id))

    candidates.sort()
    return [wall_id for _, wall_id in candidates[:max_candidates]]


def roomformer_prediction_to_layout(prediction, points, wall_thickness=0.03):
    norm = roomformer_normalization(points)
    z_min = float(np.min(points[:, 2]))
    z_max = float(np.max(points[:, 2]))
    height = max(z_max - z_min, 1e-3)
    center_z = z_min + height * 0.5

    edges = collect_edges(prediction["room_polys"], norm)
    lines = []
    for wall_id, (p1, p2) in enumerate(edges):
        lines.append(
            "wall_{id}=Wall({ax:.6f},{ay:.6f},{az:.6f},{bx:.6f},{by:.6f},{bz:.6f},{height:.6f},{thickness:.6f})".format(
                id=wall_id,
                ax=p1[0],
                ay=p1[1],
                az=z_min,
                bx=p2[0],
                by=p2[1],
                bz=z_min,
                height=height,
                thickness=wall_thickness,
            )
        )

    door_id = 0
    window_id = 0
    for opening, opening_type in zip(
        prediction["window_doors"], prediction["window_doors_types"]
    ):
        world_line = np.asarray([image_to_world_xy(p, norm) for p in opening], dtype=float)
        if opening_type == TYPE_DOOR:
            wall_ids = opening_wall_candidates(world_line, edges)
        else:
            wall_id = nearest_wall_id(world_line, edges)
            wall_ids = [] if wall_id is None else [wall_id]

        if not wall_ids:
            continue

        center = world_line.mean(axis=0)
        width = float(np.linalg.norm(world_line[1] - world_line[0]))
        if opening_type == TYPE_DOOR:
            for wall_id in wall_ids:
                wall = edges[wall_id]
                wall_center = (wall[0] + wall[1]) * 0.5
                wall_vec = wall[1] - wall[0]
                wall_len = np.linalg.norm(wall_vec)
                wall_dir = wall_vec / wall_len
                wall_pos = wall[0] + np.mean(
                    [(point - wall[0]).dot(wall_dir) for point in world_line]
                ) * wall_dir
                lines.append(
                    "door_{id}=Door(wall_{wall_id},{x:.6f},{y:.6f},{z:.6f},{width:.6f},{height:.6f})".format(
                        id=door_id,
                        wall_id=wall_id,
                        x=wall_pos[0],
                        y=wall_pos[1],
                        z=center_z,
                        width=width,
                        height=height,
                    )
                )
                door_id += 1
        elif opening_type == TYPE_WINDOW:
            wall_id = wall_ids[0]
            lines.append(
                "window_{id}=Window(wall_{wall_id},{x:.6f},{y:.6f},{z:.6f},{width:.6f},{height:.6f})".format(
                    id=window_id,
                    wall_id=wall_id,
                    x=center[0],
                    y=center[1],
                    z=center_z,
                    width=width,
                    height=height,
                )
            )
            window_id += 1

    return "\n".join(lines)


def load_roomformer_scene(scene_id, dataset_dir, prediction_dir):
    scene_name = scene_stem(scene_id)
    pred_path = Path(prediction_dir) / f"{scene_number(scene_id):05d}.json"
    point_cloud_path = Path(dataset_dir) / "pcd" / f"{scene_name}.ply"

    prediction = json.loads(pred_path.read_text())
    pcd = load_o3d_pcd(str(point_cloud_path))
    points, colors = get_points_and_colors(pcd)
    return prediction, points, colors, point_cloud_path
