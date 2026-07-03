import argparse
import math
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser("Run SceneScript batch inference in parallel across GPUs.")
    parser.add_argument("--gpus", default="0,1,2,3", help="Comma-separated GPU ids.")
    parser.add_argument("--dataset_dir", default="")
    parser.add_argument(
        "--checkpoint",
        default="baselines/SceneScript/checkpoints/scenescript_model_ase.ckpt",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene_start", type=int, default=3250)
    parser.add_argument("--scene_end", type=int, default=3500)
    parser.add_argument("--max_points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nucleus_sampling_thresh", type=float, default=0.05)
    parser.add_argument("--origin_padding", type=float, default=0.0)
    parser.add_argument("--retry_empty_seeds", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_dir", default="")
    return parser.parse_args()


def chunk_ranges(start, end, num_chunks):
    total = end - start
    chunk_size = math.ceil(total / num_chunks)
    ranges = []
    for index in range(num_chunks):
        chunk_start = start + index * chunk_size
        chunk_end = min(end, chunk_start + chunk_size)
        if chunk_start < chunk_end:
            ranges.append((chunk_start, chunk_end))
    return ranges


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    runner = repo_root / "tools" / "scenescript" / "run_batch_inference.py"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")

    log_dir = Path(args.log_dir) if args.log_dir else output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    for worker_id, (gpu, (scene_start, scene_end)) in enumerate(
        zip(gpus, chunk_ranges(args.scene_start, args.scene_end, len(gpus)))
    ):
        cmd = [
            sys.executable,
            str(runner),
            "--checkpoint",
            args.checkpoint,
            "--output_dir",
            args.output_dir,
            "--scene_start",
            str(scene_start),
            "--scene_end",
            str(scene_end),
            "--max_points",
            str(args.max_points),
            "--seed",
            str(args.seed),
            "--nucleus_sampling_thresh",
            str(args.nucleus_sampling_thresh),
            "--origin_padding",
            str(args.origin_padding),
        ]
        if args.dataset_dir:
            cmd.extend(["--dataset_dir", args.dataset_dir])
        if args.retry_empty_seeds:
            cmd.extend(["--retry_empty_seeds", args.retry_empty_seeds])
        if args.overwrite:
            cmd.append("--overwrite")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log_path = log_dir / f"worker_{worker_id}_gpu{gpu}_{scene_start}_{scene_end}.log"
        log_file = log_path.open("w")
        print(
            f"worker={worker_id} gpu={gpu} scenes=[{scene_start},{scene_end}) "
            f"log={log_path}"
        )
        process = subprocess.Popen(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((worker_id, gpu, scene_start, scene_end, process, log_file))

    failed = []
    for worker_id, gpu, scene_start, scene_end, process, log_file in processes:
        return_code = process.wait()
        log_file.close()
        if return_code != 0:
            failed.append((worker_id, gpu, scene_start, scene_end, return_code))
        print(
            f"worker={worker_id} gpu={gpu} scenes=[{scene_start},{scene_end}) "
            f"return_code={return_code}"
        )

    if failed:
        print("Failed workers:")
        for worker in failed:
            print(worker)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
