#!/usr/bin/env python3
"""Benchmark V-DETR model-forward latency with batch size one."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
VDETR_ROOT = REPO_ROOT / "baselines" / "VDETR"
sys.path.insert(0, str(VDETR_ROOT))
sys.path.insert(0, str(VDETR_ROOT / "third_party/pointnet2"))

from datasets import build_dataset  # noqa: E402
from main import auto_reload, make_args_parser  # noqa: E402
from models import build_model  # noqa: E402
from utils.dist import batch_dict_to_cuda  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=VDETR_ROOT / "checkpoints/scannet_540ep.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/model_efficiency/vdetr_raw.jsonl",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def model_args(checkpoint: Path) -> argparse.Namespace:
    args = make_args_parser().parse_args(
        [
            "--dataset_name",
            "scannet",
            "--dataset_root_dir",
            str(VDETR_ROOT / "scannet/scannet_train_detection_data"),
            "--meta_data_dir",
            str(VDETR_ROOT / "scannet/meta_data"),
            "--test_only",
            "--auto_test",
            "--test_ckpt",
            str(checkpoint),
        ]
    )
    auto_reload(args)
    args.ngpus = 1
    return args


def write_record(file, record: dict) -> None:
    file.write(json.dumps(record, sort_keys=True) + "\n")
    file.flush()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        output.unlink(missing_ok=True)

    completed = set()
    if output.exists():
        for line in output.read_text().splitlines():
            record = json.loads(line)
            if record.get("status") == "ok":
                completed.add(record["scene_id"])

    vdetr_args = model_args(checkpoint)
    datasets, dataset_config = build_dataset(vdetr_args)
    dataset = datasets["test"]
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    model = build_model(vdetr_args, dataset_config)
    model.load_state_dict(checkpoint_data["model"], strict=False)
    model = model.to(args.device).eval()
    indices = range(min(len(dataset), args.limit)) if args.limit > 0 else range(len(dataset))

    warmed_up = False
    saved = 0
    with output.open("a", encoding="utf-8") as file, torch.no_grad():
        for position, index in enumerate(indices, start=1):
            scene_id = dataset.scan_names[index]
            if scene_id in completed:
                continue

            # NPY loading, point sampling, collation, and device transfer are untimed.
            np.random.seed(zlib.crc32(scene_id.encode("utf-8")))
            batch = dataset.collate_fn([dataset[index]])
            batch = batch_dict_to_cuda(batch, local_rank=next(model.parameters()).device)
            inputs = {
                "point_clouds": batch["point_clouds"],
                "point_cloud_dims_min": batch["point_cloud_dims_min"],
                "point_cloud_dims_max": batch["point_cloud_dims_max"],
            }

            if not warmed_up:
                model(inputs)
                torch.cuda.synchronize()
                warmed_up = True

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(inputs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            point_clouds = inputs["point_clouds"]
            if torch.is_tensor(point_clouds):
                input_point_count = int(point_clouds.shape[-2])
            else:
                input_point_count = sum(int(points.shape[0]) for points in point_clouds)
            write_record(
                file,
                {
                    "model_name": "vdetr",
                    "scene_id": scene_id,
                    "status": "ok",
                    "forward_seconds": elapsed,
                    "scenes_per_second": 1.0 / elapsed,
                    "peak_memory_mb": peak_memory_mb,
                    "input_point_count": input_point_count,
                    "checkpoint": str(args.checkpoint),
                },
            )
            saved += 1
            print(
                f"[{position:03d}/{len(dataset):03d}] {scene_id}: {elapsed:.4f}s",
                flush=True,
            )
    print(f"Saved {saved} new measurements to {output}")


if __name__ == "__main__":
    main()
