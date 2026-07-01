import argparse
import csv
import shutil
import sys
from pathlib import Path


SCENESCRIPT_ROOT = Path(__file__).resolve().parents[2] / "baselines" / "SceneScript"
sys.path.insert(0, str(SCENESCRIPT_ROOT))

from src.data.language_sequence import LanguageSequence  # noqa: E402

from convert_to_spatiallm_layout import convert_language_to_spatiallm  # noqa: E402
from run_batch_inference import DEFAULT_DATASET_DIR  # noqa: E402


def scene_id_from_path(path):
    return int(path.stem.split("_")[-1])


def parse_args():
    parser = argparse.ArgumentParser(
        "Prepare SpatialLM eval files for SceneScript predictions using HF layout GT."
    )
    parser.add_argument("--prediction_dir", default="baselines/SceneScript/predictions_ase_200k")
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output_dir", default="baselines/SceneScript/spatiallm_eval_ase_200k_hfgt")
    parser.add_argument("--wall_thickness", type=float, default=0.05)
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
    missing_gt = []
    failed = []
    for prediction_path in sorted(prediction_dir.glob("scene_*.txt")):
        scene_id = scene_id_from_path(prediction_path)
        hf_gt_path = dataset_dir / "layout" / f"scene_{scene_id:05d}.txt"
        if not hf_gt_path.is_file():
            missing_gt.append(scene_id)
            continue

        try:
            language_sequence = LanguageSequence.load_from_file(prediction_path)
            pred_layout = convert_language_to_spatiallm(
                language_sequence, wall_thickness=args.wall_thickness
            )
        except Exception as exc:
            failed.append((scene_id, repr(exc)))
            continue

        (pred_dir / f"{scene_id}.txt").write_text(pred_layout)
        shutil.copyfile(hf_gt_path, gt_dir / f"{scene_id}.txt")
        scene_ids.append(scene_id)

    write_metadata(scene_ids, output_dir)
    print(f"Prepared {len(scene_ids)} scenes in {output_dir}")
    print(f"Missing GT: {len(missing_gt)}")
    if missing_gt:
        print(missing_gt)
    print(f"Failed: {len(failed)}")
    if failed:
        print(failed[:10])


if __name__ == "__main__":
    main()
