import argparse
from pathlib import Path

from roomformer_layout_utils import (
    DEFAULT_DATASET_DIR,
    load_roomformer_scene,
    roomformer_prediction_to_layout,
    scene_stem,
)


def parse_args():
    parser = argparse.ArgumentParser(
        "Convert a RoomFormer prediction JSON to a SpatialLM layout txt."
    )
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--prediction_dir",
        default="baselines/RoomFormer/checkpoints/eval_stru3d_sem_rich/predictions",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--wall_thickness", type=float, default=0.03)
    return parser.parse_args()


def main():
    args = parse_args()
    prediction, points, _, _ = load_roomformer_scene(
        args.scene_id, args.dataset_dir, args.prediction_dir
    )
    layout_str = roomformer_prediction_to_layout(
        prediction, points, wall_thickness=args.wall_thickness
    )

    output = args.output
    if output is None:
        output = f"outputs/roomformer_{scene_stem(args.scene_id)}_layout.txt"

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(layout_str)
    print(output_path)


if __name__ == "__main__":
    main()
