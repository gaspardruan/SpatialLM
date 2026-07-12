import argparse
import csv
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

from run_inference import load_points, subsample_points


SCENESCRIPT_ROOT = Path(__file__).resolve().parents[2] / "baselines" / "SceneScript"
sys.path.insert(0, str(SCENESCRIPT_ROOT))

from src.networks.scenescript_model import SceneScriptWrapper  # noqa: E402


DEFAULT_DATASET_DIR = (
    "/ssd/zq/.cache/huggingface/hub/datasets--ysmao--structured3d-spatiallm/"
    "snapshots/c5bedd45675b566547e6ae0bc077681bc58b7b35"
)


def parse_args():
    parser = argparse.ArgumentParser("Run SceneScript inference on HF Structured3D test scenes.")
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--checkpoint",
        default="baselines/SceneScript/checkpoints/scenescript_model_ase.ckpt",
    )
    parser.add_argument("--output_dir", default="baselines/SceneScript/predictions_ase_200k")
    parser.add_argument("--scene_start", type=int, default=3250)
    parser.add_argument("--scene_end", type=int, default=3500)
    parser.add_argument(
        "--metadata",
        default="",
        help="Optional metadata CSV containing scene_id and pcd columns.",
    )
    parser.add_argument("--worker_rank", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nucleus_sampling_thresh", type=float, default=0.05)
    parser.add_argument(
        "--origin_padding",
        type=float,
        default=0.0,
        help="Use 0.1 for Structured3D fine-tuned checkpoints prepared with padded origins.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--retry_empty_seeds",
        default="",
        help=(
            "Comma-separated extra seeds used only when the first inference produces "
            "0 entities, e.g. '1,2,3'. Empty by default for the original protocol."
        ),
    )
    return parser.parse_args()


def load_scene_rows(args):
    if args.metadata:
        with Path(args.metadata).open() as f:
            rows = list(csv.DictReader(f))
        return rows[args.worker_rank :: args.num_workers]

    dataset_dir = Path(args.dataset_dir)
    return [
        {
            "scene_id": f"scene_{scene_id:05d}",
            "pcd": str(dataset_dir / "pcd" / f"scene_{scene_id:05d}.ply"),
        }
        for scene_id in range(args.scene_start, args.scene_end)
    ]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_wrapper = SceneScriptWrapper.load_from_checkpoint(args.checkpoint)
    if args.device == "cuda":
        model_wrapper = model_wrapper.cuda()
    retry_empty_seeds = [
        int(seed.strip())
        for seed in args.retry_empty_seeds.split(",")
        if seed.strip()
    ]

    done = 0
    skipped = 0
    missing = []
    failed = []
    for row in load_scene_rows(args):
        scene_id = row["scene_id"]
        output_path = output_dir / f"{scene_id}.txt"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        point_cloud_path = Path(row["pcd"])
        if not point_cloud_path.is_file():
            missing.append(scene_id)
            continue

        try:
            raw_points = load_points(point_cloud_path)
            lang_seq = None
            used_seed = args.seed
            for seed in [args.seed] + retry_empty_seeds:
                scene_seed = zlib.crc32(scene_id.encode("utf-8"))
                points = subsample_points(raw_points, args.max_points, seed + scene_seed)
                if args.device == "cuda":
                    torch.cuda.empty_cache()

                candidate = model_wrapper.run_inference(
                    points,
                    nucleus_sampling_thresh=args.nucleus_sampling_thresh,
                    origin_padding=args.origin_padding,
                    verbose=False,
                )
                lang_seq = candidate
                used_seed = seed
                if len(candidate.entities) > 0:
                    break

            output_path.write_text(lang_seq.generate_language_string())
            done += 1
            retry_note = f" seed={used_seed}" if used_seed != args.seed else ""
            print(
                f"[{done:03d}] {scene_id}: "
                f"{len(lang_seq.entities)} entities{retry_note}"
            )
        except Exception as exc:
            failed.append((scene_id, repr(exc)))
            print(f"[fail] {scene_id}: {exc}")

    print(f"done={done} skipped={skipped} missing_pcd={len(missing)} failed={len(failed)}")
    if missing:
        print("missing:", missing)
    if failed:
        print("failed:", failed[:10])


if __name__ == "__main__":
    main()
