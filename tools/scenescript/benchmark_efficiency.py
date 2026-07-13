#!/usr/bin/env python3
"""Benchmark SceneScript model generation, excluding data I/O and preprocessing."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENESCRIPT_ROOT = REPO_ROOT / "baselines" / "SceneScript"
sys.path.insert(0, str(SCENESCRIPT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_inference import load_points, subsample_points  # noqa: E402
from src.networks.decoder import HELPER_TOKEN  # noqa: E402
from src.networks.scenescript_model import SceneScriptWrapper  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--scene-start", type=int, default=3250)
    parser.add_argument("--scene-end", type=int, default=3500)
    parser.add_argument("--max-points", type=int, required=True)
    parser.add_argument("--nucleus-sampling-thresh", type=float, default=0.05)
    parser.add_argument("--origin-padding", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Benchmark only the first N rows")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.metadata:
        with resolve_path(args.metadata).open(newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows or not {"scene_id", "pcd"}.issubset(rows[0]):
            raise ValueError("Metadata must contain scene_id and pcd columns")
        return rows

    if not args.dataset_dir:
        raise ValueError("Provide --metadata or --dataset-dir")
    dataset_dir = resolve_path(args.dataset_dir)
    return [
        {
            "scene_id": f"scene_{scene_id:05d}",
            "pcd": str(dataset_dir / "pcd" / f"scene_{scene_id:05d}.ply"),
        }
        for scene_id in range(args.scene_start, args.scene_end)
    ]


def set_seed(seed: int, device: str) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def generate_tokens(
    wrapper: SceneScriptWrapper,
    pc_sparse_tensor,
    pc_min,
    nucleus_sampling_thresh: float,
) -> tuple[torch.Tensor, float, float]:
    """Run encoder and autoregressive decoder and return exact generated tokens."""
    device = wrapper.device
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()

    encoded = wrapper.model["encoder"](pc_sparse_tensor)
    context = encoded["context"]
    context_mask = encoded["context_mask"]
    batch_size = context.shape[0]
    seq_value = torch.full(
        (batch_size, 1), HELPER_TOKEN.START, dtype=torch.long, device=device
    )
    seq_type = torch.full(
        (batch_size, 1), wrapper.type_token.START, dtype=torch.long, device=device
    )

    for _ in range(1, wrapper.max_num_tokens):
        logits = wrapper.model["decoder"](
            context=context,
            context_mask=context_mask,
            seq_value=seq_value,
            seq_type=seq_type,
        )
        filtered = wrapper.top_p(logits[:, -1], nucleus_sampling_thresh)
        token = torch.multinomial(F.softmax(filtered, dim=-1), 1)
        seq_value = torch.cat((seq_value, token), dim=1)
        seq_type = wrapper.type_decoding(seq_value, seq_type)
        if torch.any(seq_value[0] == HELPER_TOKEN.STOP):
            break

    if is_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024) if is_cuda else 0.0
    )
    return seq_value[0], elapsed, peak_memory_mb


def write_record(file, record: dict) -> None:
    file.write(json.dumps(record, sort_keys=True) + "\n")
    file.flush()


def main() -> None:
    args = parse_args()
    checkpoint = resolve_path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    rows = load_rows(args)
    if args.limit > 0:
        rows = rows[: args.limit]
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        output.unlink(missing_ok=True)

    completed = set()
    if output.exists():
        for line in output.read_text().splitlines():
            record = json.loads(line)
            if record.get("status") in {"ok", "missing_point_cloud"}:
                completed.add(record["scene_id"])

    wrapper = SceneScriptWrapper.load_from_checkpoint(checkpoint)
    if args.device == "cuda":
        wrapper = wrapper.cuda()

    warmed_up = False
    successes = 0
    with output.open("a", encoding="utf-8") as file:
        for index, row in enumerate(rows, start=1):
            scene_id = row["scene_id"]
            if scene_id in completed:
                continue

            point_cloud_path = resolve_path(row["pcd"])
            base = {
                "model_name": args.model_name,
                "checkpoint": str(args.checkpoint),
                "scene_id": scene_id,
                "point_cloud": str(row["pcd"]),
                "max_points": args.max_points,
                "nucleus_sampling_thresh": args.nucleus_sampling_thresh,
                "origin_padding": args.origin_padding,
            }
            if not point_cloud_path.is_file():
                write_record(file, {**base, "status": "missing_point_cloud"})
                continue

            try:
                # File I/O, subsampling, voxelization, and sparse collation are untimed.
                raw_points = load_points(point_cloud_path)
                scene_seed = args.seed + zlib.crc32(scene_id.encode("utf-8"))
                points = subsample_points(raw_points, args.max_points, scene_seed)
                pc_sparse_tensor, pc_min = wrapper.preprocess_point_cloud(
                    points, origin_padding=args.origin_padding
                )

                if not warmed_up:
                    set_seed(scene_seed, args.device)
                    generate_tokens(
                        wrapper,
                        pc_sparse_tensor,
                        pc_min,
                        args.nucleus_sampling_thresh,
                    )
                    warmed_up = True

                set_seed(scene_seed, args.device)
                seq_value, elapsed, peak_memory_mb = generate_tokens(
                    wrapper,
                    pc_sparse_tensor,
                    pc_min,
                    args.nucleus_sampling_thresh,
                )
                generated_tokens = int(seq_value.numel() - 1)  # Exclude START.

                # Language conversion is intentionally outside the timed interval.
                language = wrapper.postprocess_language(seq_value, pc_min)
                record = {
                    **base,
                    "status": "ok",
                    "raw_point_count": int(raw_points.shape[0]),
                    "input_point_count": int(points.shape[0]),
                    "generated_tokens": generated_tokens,
                    "entity_count": len(language.entities),
                    "generation_seconds": elapsed,
                    "seconds_per_token": elapsed / generated_tokens,
                    "tokens_per_second": generated_tokens / elapsed,
                    "peak_memory_mb": peak_memory_mb,
                }
                write_record(file, record)
                successes += 1
                print(
                    f"[{index:03d}/{len(rows):03d}] {scene_id}: "
                    f"{elapsed:.3f}s, {generated_tokens} tokens, "
                    f"{1000 * elapsed / generated_tokens:.2f} ms/token",
                    flush=True,
                )
            except Exception as error:
                write_record(
                    file,
                    {**base, "status": "error", "error": repr(error)},
                )
                print(f"[{index:03d}/{len(rows):03d}] {scene_id}: {error!r}", flush=True)

    print(f"Saved {successes} new scene measurements to {output}")


if __name__ == "__main__":
    main()
