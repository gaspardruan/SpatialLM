#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from prepare_spatiallm_eval_hfgt import convert_language_to_spatiallm

from src.data.language_sequence import LanguageSequence


SCANNET_CLASSES = [
    "cabinet", "bed", "chair", "sofa", "table", "door", "window",
    "bookshelf", "picture", "counter", "desk", "curtain", "refrigerator",
    "showercurtrain", "toilet", "sink", "bathtub", "garbagebin",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(Path(args.metadata).open()))

    completed = []
    failed = []
    for row in rows:
        scene_id = row["scene_id"]
        source = Path(args.prediction_dir) / f"{scene_id}.txt"
        gt = Path(args.gt_dir) / f"{scene_id}.txt"
        try:
            if source.is_file():
                sequence = LanguageSequence.load_from_file(source)
                converted = convert_language_to_spatiallm(sequence, wall_thickness=0.05)
            else:
                converted = ""
                failed.append((scene_id, f"missing prediction: {source}"))
            (pred_dir / f"{scene_id}.txt").write_text(converted)
            if not gt.is_file():
                raise FileNotFoundError(gt)
            completed.append(scene_id)
        except Exception as exc:
            failed.append((scene_id, repr(exc)))

    with (output_dir / "metadata.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id"])
        writer.writerows([[scene_id] for scene_id in completed])
    with (output_dir / "label_mapping.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["scannet18", "scannet18_eval"])
        writer.writerows([[name, name] for name in SCANNET_CLASSES])

    print(f"Prepared {len(completed)} scenes; failed={len(failed)}")
    if failed:
        print(failed[:10])


if __name__ == "__main__":
    main()
