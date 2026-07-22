#!/usr/bin/env python3
"""Benchmark RoomFormer model-forward latency with batch size one."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOMFORMER_ROOT = REPO_ROOT / "baselines" / "RoomFormer"
sys.path.insert(0, str(ROOMFORMER_ROOT))

from datasets import build_dataset  # noqa: E402
from eval import get_args_parser  # noqa: E402
from models import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOMFORMER_ROOT / "checkpoints/roomformer_stru3d_semantic_rich.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/model_efficiency/roomformer_raw.jsonl",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def roomformer_args(checkpoint: Path, device: str) -> argparse.Namespace:
    return get_args_parser().parse_args(
        [
            "--dataset_name=stru3d",
            f"--dataset_root={ROOMFORMER_ROOT / 'data/stru3d'}",
            "--eval_set=test",
            f"--checkpoint={checkpoint}",
            "--num_queries=2800",
            "--num_polys=70",
            "--semantic_classes=19",
            "--plot_pred=false",
            "--plot_density=false",
            "--plot_gt=false",
            f"--device={device}",
        ]
    )


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
        completed = {
            json.loads(line)["scene_id"]
            for line in output.read_text().splitlines()
            if line.strip() and json.loads(line).get("status") == "ok"
        }

    model_args = roomformer_args(checkpoint, args.device)
    torch.manual_seed(model_args.seed)
    np.random.seed(model_args.seed)
    random.seed(model_args.seed)
    model = build_model(model_args, train=False)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_data["model"], strict=False)
    model = model.to(args.device).eval()
    dataset = build_dataset(image_set="test", args=model_args)
    indices = range(min(len(dataset), args.limit)) if args.limit > 0 else range(len(dataset))

    warmed_up = False
    saved = 0
    with output.open("a", encoding="utf-8") as file, torch.no_grad():
        for position, index in enumerate(indices, start=1):
            # Dataset loading and device transfer are intentionally untimed.
            item = dataset[index]
            scene_id = str(item["image_id"])
            if scene_id in completed:
                continue
            samples = [item["image"].to(args.device)]

            if not warmed_up:
                model(samples)
                torch.cuda.synchronize()
                warmed_up = True

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = model(samples)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            output_elements = sum(
                value.numel() for value in outputs.values() if torch.is_tensor(value)
            )
            write_record(
                file,
                {
                    "model_name": "roomformer",
                    "scene_id": scene_id,
                    "status": "ok",
                    "forward_seconds": elapsed,
                    "scenes_per_second": 1.0 / elapsed,
                    "peak_memory_mb": peak_memory_mb,
                    "output_elements": output_elements,
                    "input_shape": list(samples[0].shape),
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
