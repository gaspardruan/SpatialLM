#!/usr/bin/env python3
"""Summarize raw SceneScript efficiency benchmark records."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", default="")
    return parser.parse_args()


def load_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.inputs)
    successful = [record for record in records if record.get("status") == "ok"]
    if not successful:
        raise RuntimeError("No successful benchmark records")

    per_scene_fields = [
        "model_name",
        "scene_id",
        "input_point_count",
        "generated_tokens",
        "entity_count",
        "generation_seconds",
        "seconds_per_token",
        "tokens_per_second",
        "peak_memory_mb",
    ]
    with (output_dir / "per_scene.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=per_scene_fields)
        writer.writeheader()
        writer.writerows(
            {field: record[field] for field in per_scene_fields}
            for record in successful
        )

    summaries = []
    for model_name in sorted({record["model_name"] for record in successful}):
        all_model_records = [
            record for record in records if record["model_name"] == model_name
        ]
        model_records = [
            record for record in successful if record["model_name"] == model_name
        ]
        times = np.asarray([record["generation_seconds"] for record in model_records])
        tokens = np.asarray([record["generated_tokens"] for record in model_records])
        memory = np.asarray([record["peak_memory_mb"] for record in model_records])
        summaries.append(
            {
                "model_name": model_name,
                "attempted_scene_count": len(all_model_records),
                "scene_count": len(model_records),
                "missing_scene_count": sum(
                    record["status"] == "missing_point_cloud"
                    for record in all_model_records
                ),
                "error_scene_count": sum(
                    record["status"] == "error" for record in all_model_records
                ),
                "max_length_scene_count": sum(
                    record["generated_tokens"] >= 999 for record in model_records
                ),
                "total_tokens": int(tokens.sum()),
                "mean_tokens_per_scene": float(tokens.mean()),
                "mean_seconds_per_scene": float(times.mean()),
                "median_seconds_per_scene": float(np.median(times)),
                "p95_seconds_per_scene": float(np.percentile(times, 95)),
                "seconds_per_token": float(times.sum() / tokens.sum()),
                "milliseconds_per_token": float(1000 * times.sum() / tokens.sum()),
                "tokens_per_second": float(tokens.sum() / times.sum()),
                "mean_peak_memory_mb": float(memory.mean()),
                "max_peak_memory_mb": float(memory.max()),
            }
        )

    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summaries, file, indent=2)
        file.write("\n")

    environment = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_index": args.gpu_index,
        "timing_scope": (
            "point-cloud encoder plus autoregressive token generation; excludes file I/O, "
            "point-cloud preprocessing, language postprocessing, and result writing"
        ),
        "token_definition": "generated sequence tokens excluding START and including STOP",
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
        "# SceneScript Efficiency",
        "",
        "Timing includes point-cloud encoding and autoregressive generation. It excludes "
        "file I/O, point-cloud preprocessing, language postprocessing, and result writing.",
        "",
        "| Model | Scenes | Mean tokens/scene | Mean s/scene | Median s/scene | P95 s/scene | ms/token | tokens/s | Peak GPU MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['model_name']} | {row['scene_count']} | "
            f"{row['mean_tokens_per_scene']:.1f} | {row['mean_seconds_per_scene']:.3f} | "
            f"{row['median_seconds_per_scene']:.3f} | {row['p95_seconds_per_scene']:.3f} | "
            f"{row['milliseconds_per_token']:.3f} | {row['tokens_per_second']:.2f} | "
            f"{row['max_peak_memory_mb']:.0f} |"
        )
    lines.extend(
        [
            "",
            "Generated-token counts exclude the initial START token and include STOP. "
            "`ms/token` is total measured generation time divided by all generated tokens.",
            "",
        ]
    )
    for row in summaries:
        lines.append(
            f"- `{row['model_name']}`: {row['missing_scene_count']} missing scenes, "
            f"{row['error_scene_count']} errors, and "
            f"{row['max_length_scene_count']} scene(s) reached the 999-token limit."
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
