import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchsparse
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from torchsparse.utils.collate import sparse_collate

from convert_spatiallm_layout_to_scenescript import load_scene_origin
from prepare_finetune_data import language_to_tokens
from run_inference import load_points, subsample_points


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


def load_training_example(
    row, cfg, max_points, seed, device, origin_padding, rotation_degrees
):
    points = load_points(row["pcd"])
    points = subsample_points(points, max_points, seed)
    origin = load_scene_origin(row["pcd"], padding_ratio=origin_padding)
    language_sequence = LanguageSequence.load_from_file(row["language"])

    points = points[:, :3] - torch.as_tensor(origin, dtype=torch.float32)
    if rotation_degrees > 0:
        rng = np.random.default_rng(seed)
        angle_degrees = float(rng.uniform(-rotation_degrees, rotation_degrees))
        angle = np.deg2rad(angle_degrees)
        rotation = torch.tensor(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        points = points @ rotation.T
        language_sequence.rotate(angle_degrees)

        point_min = points.min(dim=0).values
        point_max = points.max(dim=0).values
        shift = point_min - origin_padding * (point_max - point_min)
        points = points - shift
        language_sequence.translate(-shift)

    sparse_tensor = preprocess_point_cloud(
        points, np.zeros(3, dtype=np.float32), cfg, device
    )
    language_sequence.sort_entities("lex")
    language_sequence.normalize_and_discretize(
        cfg.data.num_bins, cfg.data.normalization_values
    )
    seq_value, seq_type = language_to_tokens(language_sequence, cfg)
    return sparse_tensor, seq_value.to(device), seq_type.to(device)


def parse_args():
    parser = argparse.ArgumentParser("Accelerate fine-tuning for SceneScript.")
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
        default="baselines/SceneScript/checkpoints/scenescript_ase_finetuned_structured3d_accelerate.ckpt",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_points", type=int, default=200000)
    parser.add_argument("--origin_padding", type=float, default=0.1)
    parser.add_argument(
        "--rotation_degrees",
        type=float,
        default=0.0,
        help="Uniform random Z rotation in [-value, value] degrees.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bbox_classes",
        default="",
        help="Comma-separated bbox taxonomy stored in the fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--distributed_backend",
        choices=["nccl", "gloo"],
        default="nccl",
        help="Distributed process-group backend. Use gloo when NCCL/NVML is broken.",
    )
    return parser.parse_args()


def get_epoch_rows(rows, epoch, seed, rank, world_size):
    epoch_rows = list(rows)
    random.Random(seed + epoch).shuffle(epoch_rows)
    remainder = len(epoch_rows) % world_size
    if remainder:
        epoch_rows.extend(epoch_rows[: world_size - remainder])
    return epoch_rows[rank::world_size]


def save_checkpoint(output, wrapper, encoder, decoder, optimizer, accelerator, step, epoch):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {}
    state_dict.update(
        {
            f"encoder.{key}": value.cpu()
            for key, value in accelerator.unwrap_model(encoder).state_dict().items()
        }
    )
    state_dict.update(
        {
            f"decoder.{key}": value.cpu()
            for key, value in accelerator.unwrap_model(decoder).state_dict().items()
        }
    )
    ckpt = {
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": wrapper.cfg,
        "step": step,
        "epoch": epoch,
    }
    accelerator.save(ckpt, output)
    latest_path = output.with_name(output.stem + "_latest.ckpt")
    accelerator.save(ckpt, latest_path)
    accelerator.print(f"Saved {output} and {latest_path}")


def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        kwargs_handlers=[InitProcessGroupKwargs(backend=args.distributed_backend)],
    )
    random.seed(args.seed + accelerator.process_index)
    np.random.seed(args.seed + accelerator.process_index)
    torch.manual_seed(args.seed + accelerator.process_index)

    rows = read_metadata(args.metadata)
    wrapper = SceneScriptWrapper.load_from_checkpoint(args.checkpoint)
    if args.bbox_classes:
        wrapper.cfg.data.normalization_values.bbox_classes = [
            value.strip() for value in args.bbox_classes.split(",") if value.strip()
        ]
    wrapper.model.train()
    encoder = wrapper.model["encoder"]
    decoder = wrapper.model["decoder"]

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    encoder, decoder, optimizer = accelerator.prepare(encoder, decoder, optimizer)

    step = 0
    for epoch in range(args.epochs):
        epoch_rows = get_epoch_rows(
            rows,
            epoch,
            args.seed,
            accelerator.process_index,
            accelerator.num_processes,
        )
        for row_idx, row in enumerate(epoch_rows):
            step += 1
            sparse_tensor, seq_value, seq_type = load_training_example(
                row,
                wrapper.cfg,
                max_points=args.max_points,
                seed=args.seed + epoch * len(rows) + accelerator.process_index + row_idx * accelerator.num_processes,
                device=accelerator.device,
                origin_padding=args.origin_padding,
                rotation_degrees=args.rotation_degrees,
            )

            with accelerator.accumulate(encoder):
                try:
                    encoded = encoder(sparse_tensor)
                except Exception as exc:
                    raise RuntimeError(
                        f"Encoder failed for scene_id={row['scene_id']} "
                        f"rank={accelerator.process_index} step={step}"
                    ) from exc
                decoder_input_value = seq_value[:-1].unsqueeze(0)
                decoder_input_type = seq_type[:-1].unsqueeze(0)
                target = seq_value[1:].unsqueeze(0)
                logits = decoder(
                    encoded["context"],
                    encoded["context_mask"],
                    decoder_input_value,
                    decoder_input_type,
                )
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    target.reshape(-1),
                )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            if step % args.log_every == 0:
                gathered_loss = accelerator.gather(loss.detach()).mean()
                accelerator.print(
                    f"epoch={epoch} step={step} row={row_idx + 1}/{len(epoch_rows)} "
                    f"loss={gathered_loss.item():.4f}"
                )

            if accelerator.is_main_process and args.save_every > 0 and step % args.save_every == 0:
                save_path = Path(args.output)
                numbered_path = save_path.with_name(f"{save_path.stem}_step{step}.ckpt")
                save_checkpoint(
                    numbered_path,
                    wrapper,
                    encoder,
                    decoder,
                    optimizer,
                    accelerator,
                    step,
                    epoch,
                )

            if args.max_steps is not None and step >= args.max_steps:
                break
        if args.max_steps is not None and step >= args.max_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(args.output, wrapper, encoder, decoder, optimizer, accelerator, step, epoch)


if __name__ == "__main__":
    main()
