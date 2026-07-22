#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-results/model_efficiency}"
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python tools/roomformer/benchmark_efficiency.py \
  --output "$OUTPUT_DIR/roomformer_raw.jsonl"

CUDA_VISIBLE_DEVICES="$GPU" \
CUDA_HOME="${CUDA_HOME:-/home/lyd/cuda-11.3}" \
LD_LIBRARY_PATH="${CUDA_HOME:-/home/lyd/cuda-11.3}/lib64:${CUDA_HOME:-/home/lyd/cuda-11.3}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}" \
baselines/VDETR/.venv/bin/python tools/vdetr/benchmark_efficiency.py \
  --output "$OUTPUT_DIR/vdetr_raw.jsonl"

.venv/bin/python tools/summarize_model_efficiency.py \
  --roomformer "$OUTPUT_DIR/roomformer_raw.jsonl" \
  --vdetr "$OUTPUT_DIR/vdetr_raw.jsonl" \
  --scenescript-summary results/scenescript_efficiency/summary.json \
  --output-dir "$OUTPUT_DIR" \
  --gpu-index "$GPU"
