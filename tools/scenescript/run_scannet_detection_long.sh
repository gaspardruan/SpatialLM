#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash tools/scenescript/run_scannet_detection_long.sh all
#   bash tools/scenescript/run_scannet_detection_long.sh train
#   bash tools/scenescript/run_scannet_detection_long.sh evaluate

STAGE="${1:-all}"
if [[ "$STAGE" != "all" && "$STAGE" != "train" && "$STAGE" != "evaluate" ]]; then
  echo "stage must be one of: all, train, evaluate" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
ACCELERATE="$ROOT/.venv/bin/accelerate"
GPU_LIST="${GPU_LIST:-0,1,2,3,5,6,7}"
NUM_GPUS="${NUM_GPUS:-7}"
CLASSES="cabinet,bed,chair,sofa,table,door,window,bookshelf,picture,counter,desk,curtain,refrigerator,showercurtrain,toilet,sink,bathtub,garbagebin"

BASE_CKPT="baselines/SceneScript/checkpoints/scenescript_pp_finetuned_scannet_inference.ckpt"
OUTPUT_CKPT="baselines/SceneScript/checkpoints/scenescript_pp_scannet_long45000.ckpt"
TRAIN_METADATA="baselines/SceneScript/scannet_finetune/train/metadata.csv"
TEST_METADATA="baselines/SceneScript/scannet_finetune/test/metadata.csv"
RUN_DIR="baselines/SceneScript/scannet_long45000"
SUMMARY="$RUN_DIR/checkpoint_metrics.tsv"

mkdir -p "$RUN_DIR"

prepare_data() {
  "$PYTHON" tools/scenescript/prepare_finetune_data.py \
    --dataset_dir data/scannet \
    --output_dir baselines/SceneScript/scannet_finetune \
    --split train \
    --checkpoint "$BASE_CKPT" \
    --origin_padding 0.1 \
    --min_extent 0.8 \
    --bbox_classes "$CLASSES"

  "$PYTHON" tools/scenescript/prepare_finetune_data.py \
    --dataset_dir data/scannet \
    --output_dir baselines/SceneScript/scannet_finetune \
    --split test \
    --checkpoint "$BASE_CKPT" \
    --origin_padding 0.1 \
    --bbox_classes "$CLASSES"
}

train_model() {
  CUDA_VISIBLE_DEVICES="$GPU_LIST" "$ACCELERATE" launch \
    --num_processes "$NUM_GPUS" --multi_gpu \
    tools/scenescript/train_finetune_accelerate.py \
    --metadata "$TRAIN_METADATA" \
    --checkpoint "$BASE_CKPT" \
    --output "$OUTPUT_CKPT" \
    --epochs 300 \
    --max_steps 45000 \
    --max_points 500000 \
    --origin_padding 0.1 \
    --rotation_degrees 180 \
    --lr 2e-5 \
    --weight_decay 1e-2 \
    --grad_accum_steps 9 \
    --log_every 100 \
    --save_every 5000 \
    --bbox_classes "$CLASSES" \
    2>&1 | tee "$RUN_DIR/train.log"

  rm -f "${OUTPUT_CKPT%.ckpt}_latest.ckpt"
}

evaluate_checkpoint() {
  local checkpoint="$1"
  local name
  name="$(basename "${checkpoint%.ckpt}")"
  local inference="$RUN_DIR/tmp_inference.ckpt"
  local predictions="$RUN_DIR/tmp_predictions"
  local evaluation="$RUN_DIR/tmp_eval"
  local log="$RUN_DIR/${name}_eval.log"

  rm -rf "$predictions" "$evaluation"
  "$PYTHON" tools/scenescript/make_inference_checkpoint.py \
    --input "$checkpoint" --output "$inference"

  "$PYTHON" tools/scenescript/run_parallel_inference.py \
    --gpus "$GPU_LIST" \
    --metadata "$TEST_METADATA" \
    --checkpoint "$inference" \
    --output_dir "$predictions" \
    --max_points 500000 \
    --seed 0 \
    --nucleus_sampling_thresh 0 \
    --origin_padding 0.1

  "$PYTHON" tools/scenescript/prepare_spatiallm_eval_scannet.py \
    --prediction_dir "$predictions" \
    --metadata "$TEST_METADATA" \
    --gt_dir data/scannet/layout \
    --output_dir "$evaluation"

  "$PYTHON" eval.py \
    --metadata "$evaluation/metadata.csv" \
    --gt_dir data/scannet/layout \
    --pred_dir "$evaluation/pred" \
    --label_mapping "$evaluation/label_mapping.tsv" \
    --label_from scannet18 \
    --label_to scannet18_eval \
    --object_classes "$CLASSES" \
    2>&1 | tee "$log"

  local metrics
  metrics="$("$PYTHON" tools/scenescript/extract_object_f1.py "$log")"
  printf '%s\t%s\n' "$checkpoint" "$metrics" >> "$SUMMARY"
}

evaluate_all() {
  printf 'checkpoint\tf1_25\tf1_50\n' > "$SUMMARY"
  shopt -s nullglob
  local checkpoints=("${OUTPUT_CKPT%.ckpt}"_step*.ckpt "$OUTPUT_CKPT")
  if (( ${#checkpoints[@]} == 0 )); then
    echo "No checkpoints found for evaluation" >&2
    exit 1
  fi
  for checkpoint in "${checkpoints[@]}"; do
    evaluate_checkpoint "$checkpoint"
  done

  local best
  best="$(awk -F '\t' 'NR > 1 {score=$2+$3; if (score>best_score || NR==2) {best_score=score; best=$1}} END {print best}' "$SUMMARY")"
  echo "Best checkpoint: $best"

  "$PYTHON" tools/scenescript/make_inference_checkpoint.py \
    --input "$best" \
    --output baselines/SceneScript/checkpoints/scenescript_model_scannet_best_inference.ckpt

  rm -rf baselines/SceneScript/predictions_scannet_best baselines/SceneScript/spatiallm_eval_scannet_best
  "$PYTHON" tools/scenescript/run_parallel_inference.py \
    --gpus "$GPU_LIST" \
    --metadata "$TEST_METADATA" \
    --checkpoint baselines/SceneScript/checkpoints/scenescript_model_scannet_best_inference.ckpt \
    --output_dir baselines/SceneScript/predictions_scannet_best \
    --max_points 500000 \
    --seed 0 \
    --nucleus_sampling_thresh 0 \
    --origin_padding 0.1
  "$PYTHON" tools/scenescript/prepare_spatiallm_eval_scannet.py \
    --prediction_dir baselines/SceneScript/predictions_scannet_best \
    --metadata "$TEST_METADATA" \
    --gt_dir data/scannet/layout \
    --output_dir baselines/SceneScript/spatiallm_eval_scannet_best

  SCENE_ID="${SCENE_ID:-scene0249_00}"
  mkdir -p outputs
  "$PYTHON" visualize.py \
    --point_cloud "data/scannet/pcd/${SCENE_ID}.ply" \
    --layout "baselines/SceneScript/spatiallm_eval_scannet_best/pred/${SCENE_ID}.txt" \
    --save "outputs/scenescript_${SCENE_ID}_best.rrd"

  rm -rf "$RUN_DIR/tmp_predictions" "$RUN_DIR/tmp_eval"
  rm -f "$RUN_DIR/tmp_inference.ckpt"
  echo "Metrics summary: $SUMMARY"
}

if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  prepare_data
  train_model
fi
if [[ "$STAGE" == "all" || "$STAGE" == "evaluate" ]]; then
  evaluate_all
fi
