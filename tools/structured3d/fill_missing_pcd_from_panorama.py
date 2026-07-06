import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOMFORMER_STRU3D = REPO_ROOT / "baselines" / "RoomFormer" / "data_preprocess" / "stru3d"
sys.path.insert(0, str(ROOMFORMER_STRU3D))

from PointCloudReaderPanorama import PointCloudReaderPanorama  # noqa: E402


DEFAULT_HF_DATASET = Path(
    "/ssd/zq/.cache/huggingface/hub/datasets--ysmao--structured3d-spatiallm/"
    "snapshots/c5bedd45675b566547e6ae0bc077681bc58b7b35"
)


def scene_name(scene_id: int) -> str:
    return f"scene_{scene_id:05d}"


def load_split_ids(split_csv: Path, split: str) -> list[int]:
    scene_ids = []
    with split_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split:
                scene_ids.append(int(row["id"].split("_")[-1]))
    return scene_ids


def missing_pcd_ids(dataset_dir: Path, split: str) -> list[int]:
    missing = []
    for scene_id in load_split_ids(dataset_dir / "split.csv", split):
        if not (dataset_dir / "pcd" / f"{scene_name(scene_id)}.ply").is_file():
            missing.append(scene_id)
    return missing


def find_scene_dir(scene_id: int, raw_roots: list[Path]) -> Path:
    name = scene_name(scene_id)
    for root in raw_roots:
        candidates = [
            root / "Structured3D" / name,
            root / name,
        ]
        for candidate in candidates:
            if (candidate / "2D_rendering").is_dir():
                return candidate
    raise FileNotFoundError(f"Could not find {name} under: {raw_roots}")


def generate_hf_compatible_pcd(scene_dir: Path, output_ply: Path, flip_y: bool = False) -> None:
    # HF Structured3D-SpatialLM matches RoomFormer/MonteFloor generation with
    # lexicographically sorted panorama sections. The order affects RGB on
    # duplicate quantized coordinates because np.unique keeps the first color.
    original_listdir = os.listdir

    def sorted_section_listdir(path):
        values = original_listdir(path)
        if str(path).endswith("2D_rendering"):
            return sorted(values)
        return values

    os.listdir = sorted_section_listdir
    try:
        reader = PointCloudReaderPanorama(
            str(scene_dir),
            random_level=0,
            generate_color=True,
            generate_normal=False,
        )
    finally:
        os.listdir = original_listdir
    points = reader.point_cloud["coords"].astype(np.float64, copy=True)
    colors = reader.point_cloud["colors"].astype(np.float64, copy=False)

    if flip_y:
        points[:, 1] *= -1.0
    points /= 1000.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output_ply), pcd, write_ascii=False):
        raise RuntimeError(f"Failed to write {output_ply}")


def point_color_records(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if len(points) == 0:
        raise ValueError(f"Empty point cloud: {path}")

    point_mm = np.rint(points * 1000.0).astype(np.int32)
    color_u8 = np.rint(np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
    records = np.empty(
        len(point_mm),
        dtype=[
            ("x", "<i4"),
            ("y", "<i4"),
            ("z", "<i4"),
            ("r", "u1"),
            ("g", "u1"),
            ("b", "u1"),
        ],
    )
    records["x"] = point_mm[:, 0]
    records["y"] = point_mm[:, 1]
    records["z"] = point_mm[:, 2]
    records["r"] = color_u8[:, 0]
    records["g"] = color_u8[:, 1]
    records["b"] = color_u8[:, 2]
    return np.sort(records, order=["x", "y", "z", "r", "g", "b"])


def print_stats(label: str, path: Path) -> None:
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    print(f"{label}: {path}")
    print(f"  points={len(points)}")
    print(f"  xyz_min={points.min(axis=0)}")
    print(f"  xyz_max={points.max(axis=0)}")
    print(f"  rgb_min={colors.min(axis=0)}")
    print(f"  rgb_max={colors.max(axis=0)}")


def validate_against_hf(generated_ply: Path, hf_ply: Path) -> bool:
    print_stats("generated", generated_ply)
    print_stats("hf", hf_ply)

    generated = point_color_records(generated_ply)
    reference = point_color_records(hf_ply)
    same = len(generated) == len(reference) and np.array_equal(generated, reference)
    print(f"exact_point_color_set_match={same}")
    if not same:
        min_len = min(len(generated), len(reference))
        mismatch = np.flatnonzero(generated[:min_len] != reference[:min_len])
        print(f"generated_count={len(generated)} hf_count={len(reference)}")
        print(f"first_sorted_mismatch_index={int(mismatch[0]) if len(mismatch) else 'count-only'}")
    return same


def copy_static_dataset_files(dataset_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["split.csv", "README.md"]:
        src = dataset_dir / name
        if src.is_file():
            shutil.copy2(src, output_dir / name)
    for subdir in ["layout"]:
        src = dataset_dir / subdir
        dst = output_dir / subdir
        if not dst.exists():
            dst.symlink_to(src, target_is_directory=True)
    pcd_dir = output_dir / "pcd"
    pcd_dir.mkdir(exist_ok=True)
    for src in sorted((dataset_dir / "pcd").glob("scene_*.ply")):
        dst = pcd_dir / src.name
        if not dst.exists():
            dst.symlink_to(src)


def parse_scene_ids(values: list[str] | None) -> list[int]:
    if not values:
        return []
    scene_ids = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            scene_ids.append(int(item.replace("scene_", "")))
    return scene_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Fill missing Structured3D-SpatialLM point clouds from raw Structured3D panorama data."
    )
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_HF_DATASET)
    parser.add_argument("--raw_root", type=Path, action="append", required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/structured3d_spatiallm_filled"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--scene_ids", nargs="*", default=None)
    parser.add_argument("--missing_from_split", action="store_true")
    parser.add_argument("--validate_scene", type=int, default=None)
    parser.add_argument("--keep_generated_validation", action="store_true")
    parser.add_argument(
        "--skip_static_copy",
        action="store_true",
        help="Only write generated PLY files; do not copy/symlink split/layout/existing pcd files.",
    )
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument(
        "--flip_y",
        action="store_true",
        help="Mirror y before writing. HF Structured3D-SpatialLM pcd files do not use this.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_roots = [root.resolve() for root in args.raw_root]
    output_dir = args.output_dir.resolve()

    if args.validate_scene is not None:
        scene_dir = find_scene_dir(args.validate_scene, raw_roots)
        generated = output_dir / "validation" / f"{scene_name(args.validate_scene)}.ply"
        generate_hf_compatible_pcd(scene_dir, generated, flip_y=args.flip_y)
        hf_ply = args.dataset_dir / "pcd" / f"{scene_name(args.validate_scene)}.ply"
        ok = validate_against_hf(generated, hf_ply)
        if ok and not args.keep_generated_validation:
            generated.unlink()
        if not ok:
            raise SystemExit(1)

    scene_ids = parse_scene_ids(args.scene_ids)
    if args.missing_from_split:
        scene_ids.extend(missing_pcd_ids(args.dataset_dir, args.split))
    scene_ids = sorted(set(scene_ids))
    if not scene_ids:
        return

    if not args.skip_static_copy:
        copy_static_dataset_files(args.dataset_dir, output_dir)
    failed = []
    for scene_id in scene_ids:
        output_ply = output_dir / "pcd" / f"{scene_name(scene_id)}.ply"
        if output_ply.is_file():
            print(f"skip existing {output_ply}")
            continue
        try:
            scene_dir = find_scene_dir(scene_id, raw_roots)
            print(f"generate {scene_name(scene_id)} from {scene_dir}")
            generate_hf_compatible_pcd(scene_dir, output_ply, flip_y=args.flip_y)
        except Exception as exc:
            failed.append((scene_id, repr(exc)))
            if not args.continue_on_error:
                raise
            print(f"failed {scene_name(scene_id)}: {exc!r}")
    if failed:
        print(f"Failed scenes: {len(failed)}")
        print(failed[:20])


if __name__ == "__main__":
    main()
