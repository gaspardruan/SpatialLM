#!/usr/bin/env python3
"""Combine feed-forward and SceneScript efficiency measurements."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import torch


PARAMETERS_M = {
    "roomformer": 42.241367,
    "scenescript_structured3d": 29.850358,
    "vdetr": 79.395388,
    "scenescript_scannet": 29.850358,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roomformer", type=Path, required=True)
    parser.add_argument("--vdetr", type=Path, required=True)
    parser.add_argument("--scenescript-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", default="")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip() and json.loads(line).get("status") == "ok"
    ]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, path in (("roomformer", args.roomformer), ("vdetr", args.vdetr)):
        records = load_jsonl(path)
        times = np.asarray([record["forward_seconds"] for record in records])
        memory = np.asarray([record["peak_memory_mb"] for record in records])
        rows.append(
            {
                "model_name": name,
                "parameters_m": PARAMETERS_M[name],
                "scene_count": len(records),
                "mean_seconds_per_scene": float(times.mean()),
                "median_seconds_per_scene": float(np.median(times)),
                "p95_seconds_per_scene": float(np.percentile(times, 95)),
                "scenes_per_second": float(len(times) / times.sum()),
                "milliseconds_per_token": "",
                "tokens_per_second": "",
                "max_peak_memory_mb": float(memory.max()),
            }
        )

    for record in json.loads(args.scenescript_summary.read_text()):
        name = record["model_name"]
        rows.append(
            {
                "model_name": name,
                "parameters_m": PARAMETERS_M[name],
                "scene_count": record["scene_count"],
                "mean_seconds_per_scene": record["mean_seconds_per_scene"],
                "median_seconds_per_scene": record["median_seconds_per_scene"],
                "p95_seconds_per_scene": record["p95_seconds_per_scene"],
                "scenes_per_second": 1.0 / record["mean_seconds_per_scene"],
                "milliseconds_per_token": record["milliseconds_per_token"],
                "tokens_per_second": record["tokens_per_second"],
                "max_peak_memory_mb": record["max_peak_memory_mb"],
            }
        )

    order = [
        "roomformer",
        "scenescript_structured3d",
        "vdetr",
        "scenescript_scannet",
    ]
    rows.sort(key=lambda row: order.index(row["model_name"]))
    fields = list(rows[0])
    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w") as file:
        json.dump(rows, file, indent=2)
        file.write("\n")

    vdetr_python = Path(__file__).resolve().parents[1] / "baselines/VDETR/.venv/bin/python"
    vdetr_environment = json.loads(
        subprocess.check_output(
            [
                str(vdetr_python),
                "-c",
                (
                    "import json, torch; "
                    "print(json.dumps({'torch_version': torch.__version__, "
                    "'torch_cuda_version': torch.version.cuda}))"
                ),
            ],
            text=True,
        )
    )
    environment = {
        "feedforward_gpu_index": args.gpu_index,
        "roomformer_environment": {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        },
        "vdetr_environment": vdetr_environment,
        "batch_size": 1,
        "timing_scope": (
            "model forward only; excludes data I/O, input preprocessing, NMS, "
            "output conversion, and result writing"
        ),
        "scenescript_environment": "results/scenescript_efficiency/environment.json",
    }
    if args.gpu_index:
        query = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={args.gpu_index}",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        environment["gpu_name"], environment["driver_version"] = [
            value.strip() for value in query.split(",", maxsplit=1)
        ]
    with (output_dir / "environment.json").open("w") as file:
        json.dump(environment, file, indent=2)
        file.write("\n")

    lines = [
        "# Model Efficiency",
        "",
        "Feed-forward timings use batch size one and exclude data I/O, input "
        "preprocessing, NMS, output conversion, and result writing.",
        "",
        "| Task | Model | State elements | Scenes | Mean s/scene | Median s/scene | P95 s/scene | Scenes/s | ms/token | Tokens/s | Peak GPU MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        task = "Layout" if row["model_name"] in {"roomformer", "scenescript_structured3d"} else "Detection"
        token_ms = f"{row['milliseconds_per_token']:.3f}" if row["milliseconds_per_token"] != "" else "N/A"
        token_rate = f"{row['tokens_per_second']:.2f}" if row["tokens_per_second"] != "" else "N/A"
        lines.append(
            f"| {task} | {row['model_name']} | {row['parameters_m']:.1f}M | "
            f"{row['scene_count']} | {row['mean_seconds_per_scene']:.3f} | "
            f"{row['median_seconds_per_scene']:.3f} | {row['p95_seconds_per_scene']:.3f} | "
            f"{row['scenes_per_second']:.2f} | {token_ms} | {token_rate} | "
            f"{row['max_peak_memory_mb']:.0f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
