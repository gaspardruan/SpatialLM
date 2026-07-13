#!/usr/bin/env python3
"""Collect final baseline predictions in SpatialLM text format."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RESULT_SETS = {
    "layout_estimation/roomformer": {
        "source": "baselines/RoomFormer/spatiallm_eval_hfgt/pred",
        "dataset": "Structured3D",
        "entities": ["Wall", "Door", "Window"],
        "checkpoint": (
            "baselines/RoomFormer/checkpoints/"
            "roomformer_stru3d_semantic_rich.pth"
        ),
    },
    "layout_estimation/scenescript_structured3d": {
        "source": (
            "baselines/SceneScript/"
            "spatiallm_eval_pp_ft_structured3d_filled2572_accelerate8_retry_empty_hfgt/pred"
        ),
        "dataset": "Structured3D",
        "entities": ["Wall", "Door", "Window"],
        "checkpoint": (
            "baselines/SceneScript/checkpoints/"
            "scenescript_model_pp_best_inference.ckpt"
        ),
    },
    "object_detection/vdetr": {
        "source": "baselines/VDETR/spatiallm_eval/pred",
        "dataset": "ScanNet",
        "entities": ["Bbox"],
        "checkpoint": "baselines/VDETR/checkpoints/scannet_540ep.pth",
    },
    "object_detection/scenescript_scannet": {
        "source": "baselines/SceneScript/spatiallm_eval_scannet_best/pred",
        "dataset": "ScanNet",
        "entities": ["Bbox"],
        "checkpoint": (
            "baselines/SceneScript/checkpoints/"
            "scenescript_model_scannet_best_inference.ckpt"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
        help="Destination directory (default: repository-root/results)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    manifest = {"format": "SpatialLM text layout", "results": {}}

    for result_name, config in RESULT_SETS.items():
        source = ROOT / config["source"]
        if not source.is_dir():
            raise FileNotFoundError(f"Missing prediction directory: {source}")

        files = sorted(source.glob("*.txt"))
        if not files:
            raise RuntimeError(f"No prediction files found in {source}")

        destination = output_dir / result_name
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        for prediction in files:
            shutil.copy2(prediction, destination / prediction.name)

        manifest["results"][result_name] = {
            "dataset": config["dataset"],
            "entities": config["entities"],
            "checkpoint": config["checkpoint"],
            "source": config["source"],
            "scene_count": len(files),
            "nonempty_scene_count": sum(file.stat().st_size > 0 for file in files),
        }
        print(f"{result_name}: {len(files)} scenes")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
