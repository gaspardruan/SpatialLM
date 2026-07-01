import argparse
import csv
import shutil
from pathlib import Path

from roomformer_layout_utils import (
    DEFAULT_DATASET_DIR,
    load_roomformer_scene,
    roomformer_prediction_to_layout,
)


def parse_args():
    parser = argparse.ArgumentParser(
        "Prepare SpatialLM eval files for RoomFormer predictions using HF layout GT."
    )
    parser.add_argument(
        "--prediction_dir",
        default="baselines/RoomFormer/checkpoints/eval_stru3d_sem_rich/predictions",
    )
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output_dir", default="baselines/RoomFormer/spatiallm_eval_hfgt")
    parser.add_argument("--wall_thickness", type=float, default=0.03)
    return parser.parse_args()


def write_metadata(scene_ids, output_dir):
    with (output_dir / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "pcd", "layout"])
        writer.writeheader()
        for scene_id in scene_ids:
            writer.writerow({"id": str(scene_id), "pcd": "", "layout": f"{scene_id}.txt"})

    (output_dir / "label_mapping.tsv").write_text("spatiallm59\tspatiallm20\n")


def main():
    args = parse_args()
    prediction_dir = Path(args.prediction_dir)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "pred"
    gt_dir = output_dir / "gt"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    for directory in (pred_dir, gt_dir):
        for old_file in directory.glob("*.txt"):
            old_file.unlink()

    scene_ids = []
    missing = []
    failed = []
    for prediction_path in sorted(prediction_dir.glob("*.json")):
        scene_id = int(prediction_path.stem)
        hf_gt_path = dataset_dir / "layout" / f"scene_{scene_id:05d}.txt"
        pcd_path = dataset_dir / "pcd" / f"scene_{scene_id:05d}.ply"
        if not hf_gt_path.is_file() or not pcd_path.is_file():
            missing.append(scene_id)
            continue

        try:
            prediction, points, _, _ = load_roomformer_scene(
                scene_id, dataset_dir, prediction_dir
            )
            pred_layout = roomformer_prediction_to_layout(
                prediction, points, wall_thickness=args.wall_thickness
            )
        except Exception as exc:
            failed.append((scene_id, repr(exc)))
            continue

        (pred_dir / f"{scene_id}.txt").write_text(pred_layout)
        shutil.copyfile(hf_gt_path, gt_dir / f"{scene_id}.txt")
        scene_ids.append(scene_id)

    write_metadata(scene_ids, output_dir)
    print(f"Prepared {len(scene_ids)} scenes in {output_dir}")
    print(f"Missing pcd/layout: {len(missing)}")
    if missing:
        print(missing)
    print(f"Failed: {len(failed)}")
    if failed:
        print(failed[:10])


if __name__ == "__main__":
    main()
