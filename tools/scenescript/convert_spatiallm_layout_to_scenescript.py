import argparse
import sys
from pathlib import Path

import numpy as np

from spatiallm import Layout
from spatiallm.pcd import get_points_and_colors, load_o3d_pcd


SCENESCRIPT_ROOT = Path(__file__).resolve().parents[2] / "baselines" / "SceneScript"
sys.path.insert(0, str(SCENESCRIPT_ROOT))

from src.data.language_sequence import LanguageSequence  # noqa: E402


DEFAULT_DATASET_DIR = (
    "/ssd/zq/.cache/huggingface/hub/datasets--ysmao--structured3d-spatiallm/"
    "snapshots/c5bedd45675b566547e6ae0bc077681bc58b7b35"
)


def fmt_float(value):
    return f"{float(value):.6f}"


def load_scene_origin(point_cloud_path, padding_ratio=0.0):
    pcd = load_o3d_pcd(str(point_cloud_path))
    points, _ = get_points_and_colors(pcd)
    points = points[:, :3]
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)
    return min_xyz - padding_ratio * (max_xyz - min_xyz)


def load_scene_extent(point_cloud_path):
    pcd = load_o3d_pcd(str(point_cloud_path))
    points, _ = get_points_and_colors(pcd)
    points = points[:, :3]
    return np.max(points, axis=0) - np.min(points, axis=0)


def convert_layout_to_scenescript(layout, pc_min=None):
    if pc_min is None:
        pc_min = np.zeros(3, dtype=float)
    pc_min = np.asarray(pc_min, dtype=float)

    lines = []
    wall_id_map = {}
    for new_id, wall in enumerate(layout.walls):
        wall_id_map[wall.id] = new_id
        lines.append(
            "make_wall, id={id}, a_x={ax}, a_y={ay}, a_z={az}, b_x={bx}, b_y={by}, b_z={bz}, height={height}, thickness={thickness}".format(
                id=new_id,
                ax=fmt_float(wall.ax - pc_min[0]),
                ay=fmt_float(wall.ay - pc_min[1]),
                az=fmt_float(wall.az - pc_min[2]),
                bx=fmt_float(wall.bx - pc_min[0]),
                by=fmt_float(wall.by - pc_min[1]),
                bz=fmt_float(wall.bz - pc_min[2]),
                height=fmt_float(wall.height),
                thickness=fmt_float(wall.thickness),
            )
        )

    for new_id, door in enumerate(layout.doors):
        wall_id = wall_id_map.get(door.wall_id)
        if wall_id is None:
            continue
        lines.append(
            "make_door, id={id}, wall0_id={wall_id}, wall1_id={wall_id}, position_x={x}, position_y={y}, position_z={z}, width={width}, height={height}".format(
                id=1000 + new_id,
                wall_id=wall_id,
                x=fmt_float(door.position_x - pc_min[0]),
                y=fmt_float(door.position_y - pc_min[1]),
                z=fmt_float(door.position_z - pc_min[2]),
                width=fmt_float(door.width),
                height=fmt_float(door.height),
            )
        )

    for new_id, window in enumerate(layout.windows):
        wall_id = wall_id_map.get(window.wall_id)
        if wall_id is None:
            continue
        lines.append(
            "make_window, id={id}, wall0_id={wall_id}, wall1_id={wall_id}, position_x={x}, position_y={y}, position_z={z}, width={width}, height={height}".format(
                id=2000 + new_id,
                wall_id=wall_id,
                x=fmt_float(window.position_x - pc_min[0]),
                y=fmt_float(window.position_y - pc_min[1]),
                z=fmt_float(window.position_z - pc_min[2]),
                width=fmt_float(window.width),
                height=fmt_float(window.height),
            )
        )

    for new_id, bbox in enumerate(layout.bboxes):
        lines.append(
            "make_bbox, id={id}, class={class_name}, position_x={x}, position_y={y}, position_z={z}, angle_z={angle}, scale_x={sx}, scale_y={sy}, scale_z={sz}".format(
                id=3000 + new_id,
                class_name=bbox.class_name.replace(" ", "_"),
                x=fmt_float(bbox.position_x - pc_min[0]),
                y=fmt_float(bbox.position_y - pc_min[1]),
                z=fmt_float(bbox.position_z - pc_min[2]),
                angle=fmt_float(bbox.angle_z),
                sx=fmt_float(bbox.scale_x),
                sy=fmt_float(bbox.scale_y),
                sz=fmt_float(bbox.scale_z),
            )
        )

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser("Convert SpatialLM layout txt to SceneScript language.")
    parser.add_argument("--layout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--point_cloud", default=None)
    parser.add_argument(
        "--translate_to_positive",
        action="store_true",
        help="Subtract the point-cloud minimum XYZ, matching SceneScript training preprocessing.",
    )
    parser.add_argument(
        "--origin_padding",
        type=float,
        default=0.1,
        help="Padding ratio applied to the point-cloud extent before choosing the scene origin.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    layout = Layout(Path(args.layout).read_text())
    pc_min = None
    if args.translate_to_positive:
        if args.point_cloud is None:
            raise ValueError("--point_cloud is required with --translate_to_positive")
        pc_min = load_scene_origin(args.point_cloud, padding_ratio=args.origin_padding)

    text = convert_layout_to_scenescript(layout, pc_min=pc_min)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)

    # Sanity check with SceneScript's own parser.
    LanguageSequence.load_from_file(output)
    print(output)


if __name__ == "__main__":
    main()
