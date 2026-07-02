import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchsparse
from torchsparse.utils.collate import sparse_collate

from prepare_finetune_data import language_to_tokens
from run_inference import load_points, subsample_points
from convert_spatiallm_layout_to_scenescript import load_scene_origin


SCENESCRIPT_ROOT = Path(__file__).resolve().parents[2] / "baselines" / "SceneScript"
sys.path.insert(0, str(SCENESCRIPT_ROOT))

from src.data.language_sequence import LanguageSequence  # noqa: E402
from src.networks.scenescript_model import SceneScriptWrapper  # noqa: E402


def read_metadata(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def preprocess_point_cloud(points, origin, cfg, device):
    points = points[:, :3] - torch.as_tensor(origin, dtype=torch.float32)
    num_bins = cfg.data.num_bins
    normalization_extent = (
        cfg.data.normalization_values.world[1] - cfg.data.normalization_values.world[0]
    )

    voxel_coords = (points / normalization_extent * num_bins).round().long()
    voxel_coords = voxel_coords.clamp(min=0, max=num_bins - 1)

    unique_coords, inverse, counts = np.unique(
        voxel_coords.numpy(), axis=0, return_inverse=True, return_counts=True
    )
    inverse = torch.as_tensor(inverse)
    counts = torch.as_tensor(counts)
    feats = torch.stack(
        [
            torch.bincount(inverse, weights=points[:, dim]) / counts
            for dim in range(points.shape[1])
        ],
        dim=1,
    )

    sparse_tensor = torchsparse.SparseTensor(
        coords=torch.as_tensor(unique_coords).int(),
        feats=feats.float(),
    )
    return sparse_collate([sparse_tensor]).to(device)


def load_training_example(row, cfg, max_points, seed, device, origin_padding):
    points = load_points(row["pcd"])
    points = subsample_points(points, max_points, seed)
    origin = load_scene_origin(row["pcd"], padding_ratio=origin_padding)
    sparse_tensor = preprocess_point_cloud(points, origin, cfg, device)

    language_sequence = LanguageSequence.load_from_file(row["language"])
    language_sequence.sort_entities("lex")
    language_sequence.normalize_and_discretize(
        cfg.data.num_bins, cfg.data.normalization_values
    )
    seq_value, seq_type = language_to_tokens(language_sequence, cfg)
    return sparse_tensor, seq_value.to(device), seq_type.to(device)


def parse_args():
    parser = argparse.ArgumentParser("Fine-tune SceneScript on prepared Structured3D data.")
    parser.add_argument(
        "--metadata",
        default="baselines/SceneScript/structured3d_finetune/train/metadata.csv",
    )
    parser.add_argument(
        "--checkpoint",
        default="baselines/SceneScript/checkpoints/scenescript_model_ase.ckpt",
    )
    parser.add_argument(
        "--output",
        default="baselines/SceneScript/checkpoints/scenescript_ase_finetuned_structured3d.ckpt",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_points", type=int, default=200000)
    parser.add_argument("--origin_padding", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def save_checkpoint(output, wrapper, optimizer, step, epoch):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": wrapper.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": wrapper.cfg,
            "step": step,
            "epoch": epoch,
        },
        output,
    )
    latest_path = output.with_name(output.stem + "_latest.ckpt")
    torch.save(
        {
            "model_state_dict": wrapper.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": wrapper.cfg,
            "step": step,
            "epoch": epoch,
        },
        latest_path,
    )
    print(f"Saved {output} and {latest_path}")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    rows = read_metadata(args.metadata)
    wrapper = SceneScriptWrapper.load_from_checkpoint(args.checkpoint)
    wrapper.model.train()
    wrapper.model.to(device)

    optimizer = torch.optim.AdamW(
        wrapper.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        random.shuffle(rows)
        for row_idx, row in enumerate(rows):
            step += 1
            sparse_tensor, seq_value, seq_type = load_training_example(
                row,
                wrapper.cfg,
                max_points=args.max_points,
                seed=args.seed + step,
                device=device,
                origin_padding=args.origin_padding,
            )

            encoded = wrapper.model["encoder"](sparse_tensor)
            decoder_input_value = seq_value[:-1].unsqueeze(0)
            decoder_input_type = seq_type[:-1].unsqueeze(0)
            target = seq_value[1:].unsqueeze(0)
            logits = wrapper.model["decoder"](
                encoded["context"],
                encoded["context_mask"],
                decoder_input_value,
                decoder_input_type,
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target.reshape(-1),
            )
            (loss / args.grad_accum_steps).backward()

            if step % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step} row={row_idx + 1}/{len(rows)} loss={loss.item():.4f}"
                )

            if args.save_every > 0 and step % args.save_every == 0:
                save_path = Path(args.output)
                numbered_path = save_path.with_name(f"{save_path.stem}_step{step}.ckpt")
                save_checkpoint(numbered_path, wrapper, optimizer, step, epoch)

            if args.max_steps is not None and step >= args.max_steps:
                break
        if args.max_steps is not None and step >= args.max_steps:
            break

    if step % args.grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    save_checkpoint(args.output, wrapper, optimizer, step, epoch)


if __name__ == "__main__":
    main()
