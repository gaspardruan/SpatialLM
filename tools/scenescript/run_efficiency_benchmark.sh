#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-.venv/bin/python}"
GPU="${GPU:-6}"
OUTPUT_DIR="${OUTPUT_DIR:-results/scenescript_efficiency}"
HF_STRUCTURED3D="${HF_STRUCTURED3D:-/ssd/zq/.cache/huggingface/hub/datasets--ysmao--structured3d-spatiallm/snapshots/c5bedd45675b566547e6ae0bc077681bc58b7b35}"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" tools/scenescript/benchmark_efficiency.py \
  --model-name scenescript_structured3d \
  --checkpoint baselines/SceneScript/checkpoints/scenescript_model_pp_best_inference.ckpt \
  --dataset-dir "$HF_STRUCTURED3D" \
  --scene-start 3250 \
  --scene-end 3500 \
  --max-points 200000 \
  --nucleus-sampling-thresh 0.05 \
  --origin-padding 0.1 \
  --output "$OUTPUT_DIR/structured3d_raw.jsonl"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" tools/scenescript/benchmark_efficiency.py \
  --model-name scenescript_scannet \
  --checkpoint baselines/SceneScript/checkpoints/scenescript_model_scannet_best_inference.ckpt \
  --metadata baselines/SceneScript/scannet_finetune/test/metadata.csv \
  --max-points 500000 \
  --nucleus-sampling-thresh 0 \
  --origin-padding 0.1 \
  --output "$OUTPUT_DIR/scannet_raw.jsonl"

"$PYTHON" tools/scenescript/summarize_efficiency.py \
  "$OUTPUT_DIR/structured3d_raw.jsonl" \
  "$OUTPUT_DIR/scannet_raw.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --gpu-index "$GPU"
