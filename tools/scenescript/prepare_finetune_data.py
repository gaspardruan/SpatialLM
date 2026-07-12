import argparse
import csv
import sys
from pathlib import Path

import torch

from convert_spatiallm_layout_to_scenescript import (
    convert_layout_to_scenescript,
    load_scene_extent,
    load_scene_origin,
)
from run_batch_inference import DEFAULT_DATASET_DIR
from spatiallm import Layout


SCENESCRIPT_ROOT = Path(__file__).resolve().parents[2] / "baselines" / "SceneScript"
sys.path.insert(0, str(SCENESCRIPT_ROOT))

from src.data.language_sequence import is_id_param, LanguageSequence  # noqa: E402
from src.networks.decoder import HELPER_TOKEN  # noqa: E402
from src.networks.scenescript_model import create_TYPE_TOKEN  # noqa: E402


def load_split_ids(split_csv, split):
    scene_ids = []
    with Path(split_csv).open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split:
                scene_ids.append(row["id"])
    return scene_ids


def scene_stem(scene_id):
    if scene_id.startswith("scene"):
        return scene_id
    return f"scene_{int(scene_id):05d}"


def language_to_tokens(language_sequence, cfg):
    type_token = create_TYPE_TOKEN()
    values = [int(HELPER_TOKEN.START)]
    types = [int(type_token.START)]

    for entity in language_sequence.entities:
        values.append(int(HELPER_TOKEN.PART))
        types.append(int(type_token.PART))

        values.append(int(entity.TOKEN + HELPER_TOKEN.NUM))
        types.append(int(type_token.COMMAND))

        for param_key in entity.PARAMS_DEFINITION:
            if is_id_param(param_key):
                continue
            values.append(int(entity.params[param_key] + HELPER_TOKEN.NUM))
            types.append(int(type_token[f"{entity.COMMAND_STRING}_{param_key}".upper()]))

    values.append(int(HELPER_TOKEN.STOP))
    types.append(int(type_token.STOP))

    max_tokens = cfg["model"]["decoder"]["max_num_tokens"]
    if len(values) > max_tokens:
        raise ValueError(f"Sequence has {len(values)} tokens, max is {max_tokens}")

    return torch.as_tensor(values, dtype=torch.long), torch.as_tensor(types, dtype=torch.long)


def validate_language_file(language_path, cfg):
    language_sequence = LanguageSequence.load_from_file(language_path)
    language_sequence.sort_entities("lex")
    language_sequence.normalize_and_discretize(
        cfg["data"]["num_bins"], cfg["data"]["normalization_values"]
    )
    return language_to_tokens(language_sequence, cfg)


def parse_args():
    parser = argparse.ArgumentParser("Prepare SceneScript fine-tuning language files.")
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output_dir", default="baselines/SceneScript/structured3d_finetune")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument(
        "--checkpoint",
        default="baselines/SceneScript/checkpoints/scenescript_model_ase.ckpt",
        help="Checkpoint used only to read the SceneScript normalization config.",
    )
    parser.add_argument("--origin_padding", type=float, default=0.1)
    parser.add_argument(
        "--min_extent",
        type=float,
        default=0.0,
        help="Skip scenes whose point-cloud span along any axis is below this value.",
    )
    parser.add_argument(
        "--bbox_classes",
        default="",
        help="Comma-separated bbox taxonomy overriding the checkpoint config.",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) / args.split
    language_dir = output_dir / "language"
    language_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["cfg"]
    if args.bbox_classes:
        cfg["data"]["normalization_values"]["bbox_classes"] = [
            value.strip() for value in args.bbox_classes.split(",") if value.strip()
        ]

    scene_ids = load_split_ids(dataset_dir / "split.csv", args.split)
    if args.limit is not None:
        scene_ids = scene_ids[: args.limit]

    rows = []
    missing = []
    failed = []
    too_thin = []
    token_lengths = []
    for scene_id in scene_ids:
        stem = scene_stem(scene_id)
        layout_path = dataset_dir / "layout" / f"{stem}.txt"
        pcd_path = dataset_dir / "pcd" / f"{stem}.ply"
        if not layout_path.is_file() or not pcd_path.is_file():
            missing.append(scene_id)
            continue

        out_path = language_dir / f"{stem}.txt"
        try:
            extent = load_scene_extent(pcd_path)
            if float(extent.min()) < args.min_extent:
                too_thin.append((scene_id, extent.tolist()))
                continue
            layout = Layout(layout_path.read_text())
            origin = load_scene_origin(pcd_path, padding_ratio=args.origin_padding)
            text = convert_layout_to_scenescript(layout, pc_min=origin)
            out_path.write_text(text)
            seq_value, _ = validate_language_file(out_path, cfg)
            token_lengths.append(int(seq_value.numel()))
        except Exception as exc:
            failed.append((scene_id, repr(exc)))
            continue

        rows.append(
            {
                "scene_id": stem,
                "pcd": str(pcd_path),
                "language": str(out_path),
            }
        )

    with (output_dir / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "pcd", "language"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Prepared {len(rows)} {args.split} scenes in {output_dir}")
    print(f"Missing pcd/layout: {len(missing)}")
    print(f"Failed: {len(failed)}")
    print(f"Below minimum extent: {len(too_thin)}")
    if too_thin:
        print(too_thin[:10])
    if failed:
        print(failed[:10])
    if token_lengths:
        print(
            "Token lengths: min={min_len} max={max_len} avg={avg_len:.1f}".format(
                min_len=min(token_lengths),
                max_len=max(token_lengths),
                avg_len=sum(token_lengths) / len(token_lengths),
            )
        )


if __name__ == "__main__":
    main()
