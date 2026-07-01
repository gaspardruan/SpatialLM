import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from spatiallm.pcd import get_points_and_colors, load_o3d_pcd


SCENESCRIPT_ROOT = Path(__file__).resolve().parents[2] / "baselines" / "SceneScript"
sys.path.insert(0, str(SCENESCRIPT_ROOT))

from src.networks.scenescript_model import SceneScriptWrapper  # noqa: E402


def load_points(point_cloud_path):
    pcd = load_o3d_pcd(str(point_cloud_path))
    points, _ = get_points_and_colors(pcd)
    return torch.as_tensor(points, dtype=torch.float32)


def subsample_points(points, max_points, seed):
    if max_points is None or points.shape[0] <= max_points:
        return points

    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def parse_args():
    parser = argparse.ArgumentParser("Run SceneScript inference on a PLY point cloud.")
    parser.add_argument("--point_cloud", required=True)
    parser.add_argument(
        "--checkpoint",
        default="baselines/SceneScript/checkpoints/scenescript_model_ase.ckpt",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--max_points",
        type=int,
        default=None,
        help="Optional random point subsampling for quick smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--nucleus_sampling_thresh",
        type=float,
        default=0.05,
        help="SceneScript nucleus sampling threshold. Official notebook uses 0.05.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    points = load_points(args.point_cloud)
    points = subsample_points(points, args.max_points, args.seed)

    if args.verbose:
        print(f"Loaded {points.shape[0]} points")
        print(f"Point cloud min: {points.min(0).values.tolist()}")
        print(f"Point cloud max: {points.max(0).values.tolist()}")

    model_wrapper = SceneScriptWrapper.load_from_checkpoint(args.checkpoint)
    if args.device == "cuda":
        model_wrapper = model_wrapper.cuda()

    lang_seq = model_wrapper.run_inference(
        points,
        nucleus_sampling_thresh=args.nucleus_sampling_thresh,
        verbose=args.verbose,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(lang_seq.generate_language_string())

    if args.verbose:
        print(f"Predicted {len(lang_seq.entities)} entities")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
