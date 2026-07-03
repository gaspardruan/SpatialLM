import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval import (  # noqa: E402
    LAYOUTS,
    Layout,
    assign_class_map,
    calc_layout_tp,
    get_entity_class,
    is_valid_dw,
    is_valid_wall,
    mean_f1,
    read_label_mapping,
)


def parse_args():
    parser = argparse.ArgumentParser("Analyze per-scene SpatialLM layout eval results.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--label_mapping", required=True)
    parser.add_argument("--label_from", default="spatiallm59")
    parser.add_argument("--label_to", default="spatiallm20")
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--top_k", type=int, default=15)
    return parser.parse_args()


def load_layouts(scene_id, args, class_map):
    with open(os.path.join(args.pred_dir, f"{scene_id}.txt"), "r") as f:
        pred_layout = Layout(f.read())
    with open(os.path.join(args.gt_dir, f"{scene_id}.txt"), "r") as f:
        gt_layout = Layout(f.read())

    pred_layout.bboxes = assign_class_map(pred_layout.bboxes, class_map)
    gt_layout.bboxes = assign_class_map(gt_layout.bboxes, class_map)

    pred_wall_lookup = {w.id: w for w in pred_layout.walls}
    gt_wall_lookup = {w.id: w for w in gt_layout.walls}
    pred_instances = list(filter(is_valid_wall, pred_layout.walls)) + list(
        filter(lambda e: is_valid_dw(e, pred_wall_lookup), pred_layout.doors + pred_layout.windows)
    )
    gt_instances = list(filter(is_valid_wall, gt_layout.walls)) + list(
        filter(lambda e: is_valid_dw(e, gt_wall_lookup), gt_layout.doors + gt_layout.windows)
    )
    return pred_instances, gt_instances, pred_wall_lookup, gt_wall_lookup


def main():
    args = parse_args()
    df = pd.read_csv(args.metadata)
    class_map = read_label_mapping(args.label_mapping, args.label_from, args.label_to)

    rows = []
    per_class = {
        threshold: defaultdict(list)
        for threshold in (0.25, 0.50)
    }
    micro = {
        threshold: {
            class_name: {"tp": 0, "pred": 0, "gt": 0}
            for class_name in LAYOUTS
        }
        for threshold in (0.25, 0.50)
    }

    for scene_id in df["id"].tolist():
        pred_instances, gt_instances, pred_wall_lookup, gt_wall_lookup = load_layouts(
            scene_id, args, class_map
        )
        for class_name in LAYOUTS:
            pred_entities = [
                entity for entity in pred_instances if get_entity_class(entity) == class_name
            ]
            gt_entities = [
                entity for entity in gt_instances if get_entity_class(entity) == class_name
            ]
            scene_row = {
                "scene_id": scene_id,
                "class": class_name,
                "num_pred": len(pred_entities),
                "num_gt": len(gt_entities),
            }
            for threshold in (0.25, 0.50):
                result = calc_layout_tp(
                    pred_entities=pred_entities,
                    gt_entities=gt_entities,
                    pred_wall_id_lookup=pred_wall_lookup,
                    gt_wall_id_lookup=gt_wall_lookup,
                    iou_threshold=threshold,
                )
                per_class[threshold][class_name].append(result)
                micro[threshold][class_name]["tp"] += result.tp
                micro[threshold][class_name]["pred"] += result.num_pred
                micro[threshold][class_name]["gt"] += result.num_gt
                suffix = "25" if threshold == 0.25 else "50"
                scene_row[f"tp_{suffix}"] = result.tp
                scene_row[f"fp_{suffix}"] = result.num_pred - result.tp
                scene_row[f"fn_{suffix}"] = result.num_gt - result.tp
                scene_row[f"f1_{suffix}"] = result.f1
            rows.append(scene_row)

    print("\nMacro F1, matching eval.py:")
    for class_name in LAYOUTS:
        print(
            f"{class_name:>6s}: "
            f"{mean_f1(per_class[0.25][class_name]):.4f} @0.25, "
            f"{mean_f1(per_class[0.50][class_name]):.4f} @0.50"
        )
    print(
        f"{'avg':>6s}: "
        f"{np.mean([mean_f1(per_class[0.25][c]) for c in LAYOUTS]):.4f} @0.25, "
        f"{np.mean([mean_f1(per_class[0.50][c]) for c in LAYOUTS]):.4f} @0.50"
    )

    print("\nMicro counts:")
    for threshold in (0.25, 0.50):
        label = "@0.25" if threshold == 0.25 else "@0.50"
        print(label)
        for class_name in LAYOUTS:
            counts = micro[threshold][class_name]
            tp = counts["tp"]
            fp = counts["pred"] - tp
            fn = counts["gt"] - tp
            precision = tp / counts["pred"] if counts["pred"] else 0.0
            recall = tp / counts["gt"] if counts["gt"] else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            print(
                f"  {class_name:>6s}: tp={tp:4d} fp={fp:4d} fn={fn:4d} "
                f"prec={precision:.3f} rec={recall:.3f} f1={f1:.3f}"
            )

    print(f"\nWorst {args.top_k} scenes per class by F1@0.50:")
    for class_name in LAYOUTS:
        class_rows = [
            row for row in rows if row["class"] == class_name and row["num_gt"] > 0
        ]
        class_rows.sort(key=lambda row: (row["f1_50"], row["tp_50"], -row["fn_50"], -row["fp_50"]))
        print(f"\n{class_name}")
        for row in class_rows[: args.top_k]:
            print(
                f"  scene_{int(row['scene_id']):05d}: "
                f"f1_50={row['f1_50']:.3f}, pred={row['num_pred']}, gt={row['num_gt']}, "
                f"tp={row['tp_50']}, fp={row['fp_50']}, fn={row['fn_50']}"
            )

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
