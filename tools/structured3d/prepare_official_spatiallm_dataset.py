import argparse
import csv
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from spatiallm.pcd import get_points_and_colors, load_o3d_pcd
from tools.roomformer.roomformer_layout_utils import (
    image_to_world_xy,
    nearest_wall_id,
    opening_wall_candidates,
    roomformer_normalization,
)


DEFAULT_HF_DATASET = Path(
    "/ssd/zq/.cache/huggingface/hub/datasets--ysmao--structured3d-spatiallm/"
    "snapshots/c5bedd45675b566547e6ae0bc077681bc58b7b35"
)


def scene_name(scene_id: int) -> str:
    return f"scene_{scene_id:05d}"


def official_split(scene_id: int) -> str:
    if scene_id < 3000:
        return "train"
    if scene_id < 3250:
        return "val"
    return "test"


def write_official_split(path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "split"])
        writer.writeheader()
        for scene_id in range(3500):
            writer.writerow({"id": scene_name(scene_id), "split": official_split(scene_id)})


def convert_lines_to_vertices(lines):
    polygons = []
    lines = np.array(lines)
    polygon = None
    while len(lines) != 0:
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, 0)

        line_id, junc_id = np.where(lines == polygon[-1])
        if len(line_id) == 0:
            break
        vertex = lines[line_id[0], 1 - junc_id[0]]
        lines = np.delete(lines, line_id, 0)

        if vertex in polygon:
            polygons.append(polygon)
            polygon = None
        else:
            polygon.append(vertex)
    return polygons


def parse_floor_plan_polys(ann):
    polygons = []
    for semantic in ann["semantics"]:
        for plane_id in semantic["planeID"]:
            if ann["planes"][plane_id]["type"] != "floor":
                continue
            line_ids = np.where(np.array(ann["planeLineMatrix"][plane_id]))[0].tolist()
            pairs = [
                np.where(np.array(ann["lineJunctionMatrix"][line_id]))[0].tolist()
                for line_id in line_ids
            ]
            poly = convert_lines_to_vertices(pairs)
            if poly:
                polygons.append((poly[0], semantic["type"]))
    return polygons


def normalize_point(point, norm):
    point = np.asarray(point, dtype=float).copy()
    min_coords = norm["min_coords"]
    max_coords = norm["max_coords"]
    image_res = norm["image_res"]
    point_2d = np.round(
        (point[:2] - min_coords[:2])
        / (max_coords[:2] - min_coords[:2])
        * image_res
    )
    point[:2] = np.minimum(np.maximum(point_2d, np.zeros_like(image_res)), image_res - 1)
    return point


def sorted_edge_key(p1, p2):
    a = tuple(np.round(p1, 5))
    b = tuple(np.round(p2, 5))
    return (a, b) if a <= b else (b, a)


def collect_walls_and_openings(ann, points):
    # RoomFormer preprocessing computed normalization on millimeter coordinates.
    norm_mm = roomformer_normalization(points * 1000.0)
    norm_m = {
        "min_coords": norm_mm["min_coords"] / 1000.0,
        "max_coords": norm_mm["max_coords"] / 1000.0,
        "image_res": norm_mm["image_res"],
    }

    junctions_img = np.array(
        [normalize_point(j["coordinate"], norm_mm) for j in ann["junctions"]]
    )

    wall_seen = set()
    walls = []
    openings = []
    for polygon, poly_type in parse_floor_plan_polys(ann):
        if poly_type == "outwall":
            continue

        img_poly = junctions_img[np.array(polygon)][:, :2]
        area = abs(Polygon(img_poly).area) if len(img_poly) >= 3 else 0.0
        if poly_type not in ["door", "window"] and area < 100:
            continue
        if poly_type in ["door", "window"] and area < 1:
            continue

        if poly_type in ["door", "window"]:
            if len(img_poly) != 4:
                continue
            mid_1 = (img_poly[0] + img_poly[1]) / 2
            mid_2 = (img_poly[1] + img_poly[2]) / 2
            mid_3 = (img_poly[2] + img_poly[3]) / 2
            mid_4 = (img_poly[3] + img_poly[0]) / 2
            if np.square(mid_1 - mid_3).sum() > np.square(mid_2 - mid_4).sum():
                opening_img = np.row_stack([mid_1, mid_3])
            else:
                opening_img = np.row_stack([mid_2, mid_4])
            opening_world = np.array(
                [image_to_world_xy(point, norm_m) for point in opening_img]
            )
            openings.append((poly_type, opening_world))
            continue

        world_poly = np.array([image_to_world_xy(point, norm_m) for point in img_poly])
        for p1, p2 in zip(world_poly, np.roll(world_poly, -1, axis=0)):
            if np.linalg.norm(p2 - p1) < 1e-6:
                continue
            key = sorted_edge_key(p1, p2)
            if key in wall_seen:
                continue
            wall_seen.add(key)
            walls.append((p1, p2))

    return walls, openings


def annotation_to_layout(annotation_path: Path, pcd_path: Path, wall_thickness: float):
    ann = json.loads(annotation_path.read_text())
    pcd = load_o3d_pcd(str(pcd_path))
    points, _ = get_points_and_colors(pcd)
    z_min = float(np.min(points[:, 2]))
    z_max = float(np.max(points[:, 2]))
    height = max(z_max - z_min, 1e-3)
    center_z = z_min + height * 0.5

    walls, openings = collect_walls_and_openings(ann, points)
    lines = []
    for wall_id, (p1, p2) in enumerate(walls):
        lines.append(
            "wall_{id}=Wall({ax},{ay},{az},{bx},{by},{bz},{height},{thickness})".format(
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
    for opening_type, opening in openings:
        width = float(np.linalg.norm(opening[1] - opening[0]))
        if opening_type == "door":
            wall_ids = opening_wall_candidates(opening, walls)
            for wall_id in wall_ids:
                p1, p2 = walls[wall_id]
                wall_vec = p2 - p1
                wall_len = np.linalg.norm(wall_vec)
                if wall_len < 1e-6:
                    continue
                wall_dir = wall_vec / wall_len
                wall_pos = p1 + np.mean([(point - p1).dot(wall_dir) for point in opening]) * wall_dir
                lines.append(
                    f"door_{door_id}=Door(wall_{wall_id},{wall_pos[0]},{wall_pos[1]},{center_z},{width},{height})"
                )
                door_id += 1
        elif opening_type == "window":
            wall_id = nearest_wall_id(opening, walls)
            if wall_id is None:
                continue
            center = opening.mean(axis=0)
            lines.append(
                f"window_{window_id}=Window(wall_{wall_id},{center[0]},{center[1]},{center_z},{width},{height})"
            )
            window_id += 1

    return "\n".join(lines)


def symlink_or_replace(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src)


def parse_args():
    parser = argparse.ArgumentParser("Prepare official 3000/250/250 Structured3D SpatialLM-style dataset.")
    parser.add_argument("--hf_dataset", type=Path, default=DEFAULT_HF_DATASET)
    parser.add_argument(
        "--annotation_root",
        type=Path,
        default=Path("/ssd/zq/datasets/Structured3D_raw/annotation_3d/Structured3D"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/structured3d_spatiallm_official"),
    )
    parser.add_argument("--wall_thickness", type=float, default=0.0)
    parser.add_argument("--scene_start", type=int, default=0)
    parser.add_argument("--scene_end", type=int, default=3500)
    parser.add_argument("--overwrite_generated_layout", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    pcd_dir = output_dir / "pcd"
    layout_dir = output_dir / "layout"
    pcd_dir.mkdir(parents=True, exist_ok=True)
    layout_dir.mkdir(parents=True, exist_ok=True)
    write_official_split(output_dir / "split.csv")

    for src in sorted((args.hf_dataset / "pcd").glob("scene_*.ply")):
        symlink_or_replace(src, pcd_dir / src.name)
    for src in sorted((args.hf_dataset / "layout").glob("scene_*.txt")):
        symlink_or_replace(src, layout_dir / src.name)

    generated = []
    missing_pcd = []
    missing_annotation = []
    failed = []
    for scene_id in range(args.scene_start, args.scene_end):
        name = scene_name(scene_id)
        layout_path = layout_dir / f"{name}.txt"
        pcd_path = pcd_dir / f"{name}.ply"
        annotation_path = args.annotation_root / name / "annotation_3d.json"
        if layout_path.exists() and not args.overwrite_generated_layout:
            continue
        if not pcd_path.exists():
            missing_pcd.append(scene_id)
            continue
        if not annotation_path.is_file():
            missing_annotation.append(scene_id)
            continue
        try:
            layout = annotation_to_layout(annotation_path, pcd_path, args.wall_thickness)
            layout_path.write_text(layout)
            generated.append(scene_id)
        except Exception as exc:
            failed.append((scene_id, repr(exc)))

    def count_missing(subdir, suffix):
        return [
            scene_id
            for scene_id in range(3500)
            if not (output_dir / subdir / f"{scene_name(scene_id)}.{suffix}").exists()
        ]

    print(f"Generated layouts: {len(generated)}")
    print(f"Missing pcd while generating layout: {len(missing_pcd)}")
    print(f"Missing annotation: {len(missing_annotation)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(failed[:10])
    print(f"Official missing pcd: {len(count_missing('pcd', 'ply'))}")
    print(f"Official missing layout: {len(count_missing('layout', 'txt'))}")


if __name__ == "__main__":
    main()
