# SpatialLM

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="figures/logo_light.png#gh-light-mode-only" width="60%" alt="SpatialLM" />
  <img src="figures/logo_dark.png#gh-dark-mode-only" width="60%" alt="SpatialLM" />
</div>
<hr style="margin-top: 0; margin-bottom: 8px;">
<div align="center" style="margin-top: 0; padding-top: 0; line-height: 1;">
    <a href="https://manycore-research.github.io/SpatialLM" target="_blank" style="margin: 2px;"><img alt="Project"
    src="https://img.shields.io/badge/🌐%20Website-SpatialLM-ffc107?color=42a5f5&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
    <a href="https://arxiv.org/abs/2506.07491" target="_blank" style="margin: 2px;"><img alt="arXiv"
    src="https://img.shields.io/badge/arXiv-Techreport-b31b1b?logo=arxiv&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
    <a href="https://github.com/manycore-research/SpatialLM" target="_blank" style="margin: 2px;"><img alt="GitHub"
    src="https://img.shields.io/badge/GitHub-SpatialLM-24292e?logo=github&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
</div>
<div align="center" style="line-height: 1;">
    <a href="https://huggingface.co/manycore-research/SpatialLM1.1-Qwen-0.5B" target="_blank" style="margin: 2px;"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SpatialLM-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
    <a href="https://huggingface.co/datasets/manycore-research/SpatialLM-Dataset" target="_blank" style="margin: 2px;"><img alt="Dataset"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Dataset-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
    <a href="https://huggingface.co/datasets/manycore-research/SpatialLM-Testset" target="_blank" style="margin: 2px;"><img alt="Dataset"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Testset-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
</div>

## ✨ News

- [Sept, 2025] [SpatialLM-Dataset](https://huggingface.co/datasets/manycore-research/SpatialLM-Dataset) is now available on Hugging Face.
- [Sept, 2025] SpatialLM accepted at NeurIPS 2025.
- [Jun, 2025] Added finetuning instructions in [FINETUNE.md](./FINETUNE.md).
- [Jun, 2025] Check out our new models: [SpatialLM1.1-Llama-1B](https://huggingface.co/manycore-research/SpatialLM1.1-Llama-1B) and [SpatialLM1.1-Qwen-0.5B](https://huggingface.co/manycore-research/SpatialLM1.1-Qwen-0.5B), now available on Hugging Face. SpatialLM1.1 doubles the point cloud resolution, incorporates a more powerful point cloud encoder [Sonata](https://xywu.me/sonata/) and supports detection with user-specified categories.
- [Jun, 2025] SpatialLM [Technical Report](https://arxiv.org/abs/2506.07491) is now on arXiv.
- [Mar, 2025] We're excited to release the [SpatialLM-Llama-1B](https://huggingface.co/manycore-research/SpatialLM-Llama-1B) and [SpatialLM-Qwen-0.5B](https://huggingface.co/manycore-research/SpatialLM-Qwen-0.5B) on Hugging Face.
- [Mar, 2025] Initial release of SpatialLM!

## Introduction

SpatialLM is a 3D large language model designed to process 3D point cloud data and generate structured 3D scene understanding outputs. These outputs include architectural elements like walls, doors, windows, and oriented object bounding boxes with their semantic categories. Unlike previous methods that require specialized equipment for data collection, SpatialLM can handle point clouds from diverse sources such as monocular video sequences, RGBD images, and LiDAR sensors. This multimodal architecture effectively bridges the gap between unstructured 3D geometric data and structured 3D representations, offering high-level semantic understanding. It enhances spatial reasoning capabilities for applications in embodied robotics, autonomous navigation, and other complex 3D scene analysis tasks.

<div align="center">
  <video src="https://github.com/user-attachments/assets/c0218d6a-f676-41f8-ae76-bba228866306" poster="figures/cover.png"> </video>
  <p><i>SpatialLM reconstructs 3D layout from a monocular RGB video with MASt3R-SLAM. Results aligned to video with GT cameras for visualization.</i></p>
</div>

## SpatialLM Models

<div align="center">

|       **Model**        | **Download**                                                                      |
| :--------------------: | --------------------------------------------------------------------------------- |
| SpatialLM1.1-Llama-1B  | [🤗 HuggingFace](https://huggingface.co/manycore-research/SpatialLM1.1-Llama-1B)  |
| SpatialLM1.1-Qwen-0.5B | [🤗 HuggingFace](https://huggingface.co/manycore-research/SpatialLM1.1-Qwen-0.5B) |
| SpatialLM1.0-Llama-1B  | [🤗 HuggingFace](https://huggingface.co/manycore-research/SpatialLM-Llama-1B)     |
| SpatialLM1.0-Qwen-0.5B | [🤗 HuggingFace](https://huggingface.co/manycore-research/SpatialLM-Qwen-0.5B)    |

</div>

## Usage

### Installation

Tested with the following environment:

- Python 3.11
- Pytorch 2.4.1
- CUDA Version 12.4

```bash
# clone the repository
git clone https://github.com/manycore-research/SpatialLM.git
cd SpatialLM

# create a conda environment with cuda 12.4
conda create -n spatiallm python=3.11
conda activate spatiallm
conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit conda-forge::sparsehash

# Install dependencies with poetry
pip install poetry && poetry config virtualenvs.create false --local
poetry install
# SpatialLM1.0 dependency
poe install-torchsparse # Building wheel for torchsparse will take a while
# SpatialLM1.1 dependency
poe install-sonata # Building wheel for flash-attn will take a while
```

### Inference

In the current version of SpatialLM, input point clouds are considered axis-aligned where the z-axis is the up axis. This orientation is crucial for maintaining consistency in spatial understanding and scene interpretation across different datasets and applications.
Example preprocessed point clouds, reconstructed from RGB videos using [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM), are available in [SpatialLM-Testset](#spatiallm-testset).

Download an example point cloud:

```bash
huggingface-cli download manycore-research/SpatialLM-Testset pcd/scene0000_00.ply --repo-type dataset --local-dir .
```

Run inference:

```bash
python inference.py --point_cloud pcd/scene0000_00.ply --output scene0000_00.txt --model_path manycore-research/SpatialLM1.1-Qwen-0.5B
```

### Detection with user-specified categories

SpatialLM1.1 supports object detection conditioned on user-specified categories by leveraging the flexibility of LLMs.

SpatialLM1.1 offers three variants of structured indoor modeling tasks:

- **Structured Reconstruction**: Detect walls, doors, windows, boxes.
- **Layout Estimation**: Detect walls, doors, windows.
- **3D Object Detection**: Detect boxes.

For tasks that include object box estimation, you can specify a subset of the 59 furniture categories, and the model will only predict objects within those specified categories. For example:

```bash
python inference.py --point_cloud pcd/scene0000_00.ply --output scene0000_00.txt --model_path manycore-research/SpatialLM1.1-Qwen-0.5B --detect_type object --category bed nightstand
```

### Visualization

Use `rerun` to visualize the point cloud and the predicted structured 3D layout output:

```bash
# Convert the predicted layout to Rerun format
python visualize.py --point_cloud pcd/scene0000_00.ply --layout scene0000_00.txt --save scene0000_00.rrd

# Visualize the point cloud and the predicted layout
rerun scene0000_00.rrd
```

### Evaluation

To evaluate the performance of SpatialLM, we provide `eval.py` script that reports the benchmark results on the SpatialLM-Testset in the table below in section [Benchmark Results](#benchmark-results).

Download the testset:

```bash
huggingface-cli download manycore-research/SpatialLM-Testset --repo-type dataset --local-dir SpatialLM-Testset
```

Run evaluation:

```bash
# Run inference on the PLY point clouds in folder SpatialLM-Testset/pcd with SpatialLM1.1-Qwen-0.5B model
python inference.py --point_cloud SpatialLM-Testset/pcd --output SpatialLM-Testset/pred --model_path manycore-research/SpatialLM1.1-Qwen-0.5B

# Evaluate the predicted layouts
python eval.py --metadata SpatialLM-Testset/test.csv --gt_dir SpatialLM-Testset/layout --pred_dir SpatialLM-Testset/pred --label_mapping SpatialLM-Testset/benchmark_categories.tsv
```

### Example using a custom video

We provide an example of how to use our model to estimate scene layout starting from a RGB video with the newly released [SLAM3R](https://github.com/PKU-VCL-3DV/SLAM3R) in [EXAMPLE.md](EXAMPLE.md). These steps work for MASt3R-SLAM, and other reconstruction methods as well.

### Finetune on Custom Data

For instructions on fine-tuning SpatialLM on your own data, please refer to [FINETUNE.md](./FINETUNE.md). We provide an example using the [ARKitScenes](https://github.com/apple/ARKitScenes) dataset.

## SpatialLM Dataset

The SpatialLM dataset is a large-scale, high-quality synthetic dataset designed by professional 3D designers and used for real-world production. It contains point clouds from 12,328 diverse indoor scenes comprising 54,778 rooms, each paired with rich ground-truth 3D annotations. SpatialLM dataset provides an additional valuable resource for advancing research in indoor scene understanding, 3D perception, and related applications.

For access to photorealistic RGB/Depth/Normal/Semantic/Instance panoramic renderings and camera trajectories used to generate the SpatialLM point clouds, please refer to the [SpatialGen project](https://manycore-research.github.io/SpatialGen) for more details.

<div align="center">

|    **Dataset**    | **Download**                                                                       |
| :---------------: | ---------------------------------------------------------------------------------- |
| SpatialLM-Dataset | [🤗 Datasets](https://huggingface.co/datasets/manycore-research/SpatialLM-Dataset) |

</div>

## SpatialLM Testset

We provide a test set of 107 preprocessed point clouds, reconstructed from RGB videos using [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM). SpatialLM-Testset is quite challenging compared to prior clean RGBD scans datasets due to the noises and occlusions in the point clouds reconstructed from monocular RGB videos.

<div align="center">

|    **Dataset**    | **Download**                                                                       |
| :---------------: | ---------------------------------------------------------------------------------- |
| SpatialLM-Testset | [🤗 Datasets](https://huggingface.co/datasets/manycore-research/SpatialLM-TestSet) |

</div>

## Benchmark Results

### Layout Estimation

Layout estimation focuses on predicting architectural elements, i.e., walls, doors, and windows, within an indoor scene. We evaluated this task on the [Structured3D](https://structured3d-dataset.org) dataset. For [RoomFormer](https://github.com/ywyue/RoomFormer), we directly downloaded the model checkpoint. SceneScript and SpatialLM were first trained on our dataset, and further fine-tuned on Structured3D.

We thank @chinmay0301ucsd for identifying and fixing a bug [#88](https://github.com/manycore-research/SpatialLM/pull/88) in the evaluation script that affected door and window metrics. As a result, the scores are higher than previously reported.

<div align="center">

|   **Method**    | **RoomFormer** | **SceneScript (finetuned)** | **SpatialLM1.1-Qwen-0.5B (finetuned)** |
| :-------------: | :------------: | :-------------------------: | :------------------------------------: |
| **F1 @.25 IoU** |      83.4      |            90.4             |                  94.3                  |
| **F1 @.5 IoU**  |      81.4      |            89.2             |                  93.5                  |

</div>

#### Reproducing the RoomFormer baseline

The RoomFormer baseline is vendored under `baselines/RoomFormer`. The semantically-rich Structured3D evaluation predicts rooms, doors, and windows, and writes both aggregate metrics and per-scene prediction JSON files.

Prepare the RoomFormer dependency group and compile the two CUDA extensions:

```bash
# From the SpatialLM repository root
poetry install --with roomformer

cd baselines/RoomFormer/models/ops
sh make.sh
python test.py  # optional sanity check for MultiScaleDeformableAttention

cd ../../diff_ras
python -m pip install -e . --no-build-isolation

cd ../../..
```

The RoomFormer data and checkpoint should follow the original RoomFormer layout:

```text
baselines/RoomFormer/
├── data/stru3d/
│   ├── test/
│   └── annotations/test.json
├── checkpoints/roomformer_stru3d_semantic_rich.pth
└── s3d_floorplan_eval/montefloor_data/
```

Run the semantically-rich Structured3D evaluation:

```bash
cd baselines/RoomFormer
chmod +x tools/eval_stru3d_sem_rich.sh
CUDA_VISIBLE_DEVICES=1 ./tools/eval_stru3d_sem_rich.sh
cd ../..
```

The main outputs are:

```text
baselines/RoomFormer/checkpoints/eval_stru3d_sem_rich/results.txt
baselines/RoomFormer/checkpoints/eval_stru3d_sem_rich/predictions/*.json
```

To reproduce the RoomFormer numbers in the SpatialLM layout-estimation table, convert the RoomFormer semantically-rich predictions to the same text layout format consumed by `eval.py`. The prediction coordinates are projected back into the Hugging Face Structured3D point-cloud frame, and the GT is copied from the Hugging Face `layout/` files:

```bash
python tools/roomformer/prepare_spatiallm_eval_hfgt.py \
  --prediction_dir baselines/RoomFormer/checkpoints/eval_stru3d_sem_rich/predictions \
  --output_dir baselines/RoomFormer/spatiallm_eval_hfgt
```

This produces the table-evaluation inputs:

```text
baselines/RoomFormer/spatiallm_eval_hfgt/
├── metadata.csv
├── label_mapping.tsv
├── gt/*.txt
└── pred/*.txt
```

Then run the SpatialLM evaluator on the converted RoomFormer outputs:

```bash
python eval.py \
  --metadata baselines/RoomFormer/spatiallm_eval_hfgt/metadata.csv \
  --gt_dir baselines/RoomFormer/spatiallm_eval_hfgt/gt \
  --pred_dir baselines/RoomFormer/spatiallm_eval_hfgt/pred \
  --label_mapping baselines/RoomFormer/spatiallm_eval_hfgt/label_mapping.tsv
```

For example, after evaluating scene `03250`, visualize the RoomFormer prediction with the corresponding SpatialLM Structured3D point cloud:

```bash
python tools/roomformer/convert_prediction_to_spatiallm_layout.py \
  --scene_id 03250 \
  --output outputs/roomformer_scene_03250_layout_hfgt_fixed_doors.txt

python visualize.py \
  --point_cloud /ssd/zq/.cache/huggingface/hub/datasets--ysmao--structured3d-spatiallm/snapshots/c5bedd45675b566547e6ae0bc077681bc58b7b35/pcd/scene_03250.ply \
  --layout outputs/roomformer_scene_03250_layout_hfgt_fixed_doors.txt \
  --save outputs/roomformer_scene_03250_hfgt_fixed_doors.rrd

rerun outputs/roomformer_scene_03250_hfgt_fixed_doors.rrd --web-viewer --renderer webgl
```

If the Rerun viewer is upgraded, regenerate the `.rrd` files with the same `rerun-sdk` version that will be used to open them.

### 3D Object Detection

We evaluate 3D object detection on [ScanNet](http://www.scan-net.org) with annotations of 18 object categories. For [V-DETR](https://github.com/V-DETR/V-DETR), we directly download the model checkpoint. SceneScript and SpatialLM were first trained on our dataset, and further fine-tuned on ScanNet.

<div align="center">

|   **Method**    | **V-DETR** | **SceneScript (finetuned)** | **SpatialLM1.1-Qwen-0.5B (finetuned)** |
| :-------------: | :--------: | :-------------------------: | :------------------------------------: |
| **F1 @.25 IoU** |    65.1    |            49.1             |                  65.6                  |
| **F1 @.5 IoU**  |    56.8    |            36.8             |                  52.6                  |

</div>

#### Reproducing the V-DETR baseline

V-DETR uses its own Python 3.8 environment under `baselines/VDETR/.venv`. Prepare
the SpatialLM evaluation metadata once:

```bash
cd baselines/VDETR
.venv/bin/python tools/prepare_spatiallm_eval.py \
  --split_csv ../../data/scannet/split.csv \
  --output_dir spatiallm_eval
```

Export predictions with independent `0.5` objectness and semantic thresholds,
followed by class-aware 3D NMS:

```bash
CUDA_VISIBLE_DEVICES=1 \
CUDA_HOME=/home/lyd/cuda-11.3 \
LD_LIBRARY_PATH=/home/lyd/cuda-11.3/lib64:/home/lyd/cuda-11.3/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
OMP_NUM_THREADS=12 \
.venv/bin/python main.py \
  --dataset_name scannet \
  --dataset_root_dir scannet/scannet_train_detection_data/ \
  --meta_data_dir scannet/meta_data/ \
  --test_only --auto_test \
  --test_ckpt checkpoints/scannet_540ep.pth \
  --spatiallm_pred_dir spatiallm_eval/pred \
  --spatiallm_objectness_threshold 0.5 \
  --spatiallm_semantic_threshold 0.5 \
  --spatiallm_export_only
```

Run the SpatialLM per-scene F1 evaluation from the repository root:

```bash
python eval.py \
  --metadata baselines/VDETR/spatiallm_eval/metadata.csv \
  --gt_dir data/scannet/layout \
  --pred_dir baselines/VDETR/spatiallm_eval/pred \
  --label_mapping baselines/VDETR/spatiallm_eval/label_mapping.tsv \
  --label_from scannet18 \
  --label_to scannet18_eval \
  --object_classes cabinet,bed,chair,sofa,table,door,window,bookshelf,picture,counter,desk,curtain,refrigerator,showercurtrain,toilet,sink,bathtub,garbagebin
```

The current public checkpoint and code produce `68.0` F1 at IoU 0.25 and `61.3`
at IoU 0.5. The paper reports `65.1` and `56.8`; its V-DETR-to-SpatialLM
conversion and exact NMS configuration were not released.

Visualize a V-DETR prediction with its axis-aligned ScanNet point cloud. Run
these commands from the repository root:

```bash
SCENE_ID=scene0249_00
mkdir -p outputs

.venv/bin/python visualize.py \
  --point_cloud data/scannet/pcd/${SCENE_ID}.ply \
  --layout baselines/VDETR/spatiallm_eval/pred/${SCENE_ID}.txt \
  --save outputs/vdetr_${SCENE_ID}.rrd \
  --max_points 1000000 \
  --radius 0.01

.venv/bin/rerun outputs/vdetr_${SCENE_ID}.rrd --web-viewer
```

Change `SCENE_ID` to any scene listed in
`baselines/VDETR/scannet/meta_data/scannetv2_val.txt`. The PLY point cloud and
V-DETR prediction use the same ScanNet `axisAlignment` coordinate frame.

#### Reproducing the SceneScript detection baseline

Prepare the ScanNet `make_bbox` training and test sequences from the repository
root. The minimum-extent filter removes two incomplete training scans that
collapse to an empty tensor in SceneScript's four-stage sparse encoder.

```bash
SCANNET_CLASSES=cabinet,bed,chair,sofa,table,door,window,bookshelf,picture,counter,desk,curtain,refrigerator,showercurtrain,toilet,sink,bathtub,garbagebin

python tools/scenescript/prepare_finetune_data.py \
  --dataset_dir data/scannet \
  --output_dir baselines/SceneScript/scannet_finetune \
  --split train \
  --checkpoint baselines/SceneScript/checkpoints/scenescript_model_pp.ckpt \
  --origin_padding 0.1 \
  --min_extent 0.8 \
  --bbox_classes "$SCANNET_CLASSES"

python tools/scenescript/prepare_finetune_data.py \
  --dataset_dir data/scannet \
  --output_dir baselines/SceneScript/scannet_finetune \
  --split test \
  --checkpoint baselines/SceneScript/checkpoints/scenescript_model_pp.ckpt \
  --origin_padding 0.1 \
  --bbox_classes "$SCANNET_CLASSES"
```

Fine-tune on seven GPUs. GPU 4 is intentionally excluded on this machine.
Rotation augmentation matches the SceneScript training protocol, and gradient
accumulation gives an effective batch size of approximately 63.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 \
.venv/bin/accelerate launch --num_processes 7 --multi_gpu \
  tools/scenescript/train_finetune_accelerate.py \
  --metadata baselines/SceneScript/scannet_finetune/train/metadata.csv \
  --checkpoint baselines/SceneScript/checkpoints/scenescript_model_pp.ckpt \
  --output baselines/SceneScript/checkpoints/scenescript_pp_finetuned_scannet_accelerate7.ckpt \
  --epochs 100 \
  --max_steps 15000 \
  --max_points 200000 \
  --origin_padding 0.1 \
  --rotation_degrees 180 \
  --lr 1e-4 \
  --weight_decay 1e-2 \
  --grad_accum_steps 9 \
  --save_every 0 \
  --bbox_classes "$SCANNET_CLASSES"

python tools/scenescript/make_inference_checkpoint.py \
  --input baselines/SceneScript/checkpoints/scenescript_pp_finetuned_scannet_accelerate7.ckpt \
  --output baselines/SceneScript/checkpoints/scenescript_pp_finetuned_scannet_inference.ckpt
```

Run greedy inference, convert the predictions, and evaluate all 312 test scenes:

```bash
python tools/scenescript/run_parallel_inference.py \
  --gpus 0,1,2,3,5,6,7 \
  --metadata baselines/SceneScript/scannet_finetune/test/metadata.csv \
  --checkpoint baselines/SceneScript/checkpoints/scenescript_pp_finetuned_scannet_inference.ckpt \
  --output_dir baselines/SceneScript/predictions_pp_ft_scannet \
  --max_points 200000 \
  --nucleus_sampling_thresh 0 \
  --origin_padding 0.1

python tools/scenescript/prepare_spatiallm_eval_scannet.py \
  --prediction_dir baselines/SceneScript/predictions_pp_ft_scannet \
  --metadata baselines/SceneScript/scannet_finetune/test/metadata.csv \
  --gt_dir data/scannet/layout \
  --output_dir baselines/SceneScript/spatiallm_eval_pp_ft_scannet

python eval.py \
  --metadata baselines/SceneScript/spatiallm_eval_pp_ft_scannet/metadata.csv \
  --gt_dir data/scannet/layout \
  --pred_dir baselines/SceneScript/spatiallm_eval_pp_ft_scannet/pred \
  --label_mapping baselines/SceneScript/spatiallm_eval_pp_ft_scannet/label_mapping.tsv \
  --label_from scannet18 \
  --label_to scannet18_eval \
  --object_classes "$SCANNET_CLASSES"
```

This 15,000-step run obtains `35.83` F1 at IoU 0.25 and `27.31` at IoU 0.5.
The paper reports `49.1` and `36.8` using the original SceneScript training
budget of approximately 200,000 iterations over 3-4 days.

Visualize a SceneScript detection:

```bash
SCENE_ID=scene0249_00
.venv/bin/python visualize.py \
  --point_cloud data/scannet/pcd/${SCENE_ID}.ply \
  --layout baselines/SceneScript/spatiallm_eval_pp_ft_scannet/pred/${SCENE_ID}.txt \
  --save outputs/scenescript_${SCENE_ID}.rrd

.venv/bin/rerun outputs/scenescript_${SCENE_ID}.rrd --web-viewer
```

### Zero-shot Detection on Videos

Zero-shot detection results on the challenging SpatialLM-Testset are reported in the following table:

<div align="center">

|   **Method**    | **SpatialLM1.1-Llama-1B** | **SpatialLM1.1-Qwen-0.5B** |
| :-------------: | :-----------------------: | :------------------------: |
|   **Layout**    |   **F1 @.25 IoU (2D)**    |    **F1 @.25 IoU (2D)**    |
|      wall       |           68.9            |            68.2            |
|      door       |           49.1            |            47.4            |
|     window      |           47.0            |            51.4            |
|                 |                           |                            |
|   **Objects**   |   **F1 @.25 IoU (3D)**    |    **F1 @.25 IoU (2D)**    |
|     curtain     |           34.9            |            37.0            |
|   nightstand    |           62.8            |            67.0            |
|   chandelier    |           53.5            |            36.8            |
|    wardrobe     |           29.4            |            39.6            |
|       bed       |           96.8            |            95.2            |
|      sofa       |           66.9            |            69.1            |
|      chair      |           20.8            |            32.3            |
|     cabinet     |           15.2            |            11.2            |
|  dining table   |           40.7            |            24.2            |
|     plants      |           29.5            |            26.3            |
|   tv cabinet    |           34.4            |            27.3            |
|  coffee table   |           56.4            |            64.9            |
|   side table    |           14.6            |            9.7             |
| air conditioner |           16.7            |            24.0            |
|     dresser     |           46.7            |            46.7            |
|      stool      |           17.6            |            30.8            |
|  refrigerator   |            0.0            |            16.7            |
|    painting     |           34.9            |            38.2            |
|     carpet      |           40.3            |            24.1            |
|       tv        |           16.0            |            18.0            |

</div>

### Result Visualizations

<div align="center">

|                                                            Layout Estimation                                                            |                                                          Object Detection                                                          |                                                       Zero-shot Reconstruction                                                        |
| :-------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------: |
|                                                  ![Structured3D](./figures/stru3d.jpg)                                                  |                                                 ![ScanNet](./figures/scannet.jpg)                                                  |                                                 ![Zero-shot](./figures/zeroshot.jpg)                                                  |
| [Structured3D Results](https://manycore-research-azure.kujiale.com/manycore-research/SpatialLM/supplementary/visualization_layout.html) | [ScanNet Results](https://manycore-research-azure.kujiale.com/manycore-research/SpatialLM/supplementary/visualization_object.html) | [Zeroshot Results](https://manycore-research-azure.kujiale.com/manycore-research/SpatialLM/supplementary/visualization_zeroshot.html) |

</div>

## License

SpatialLM-Llama-1B is derived from Llama3.2-1B-Instruct, which is licensed under the Llama3.2 license.
SpatialLM-Qwen-0.5B is derived from the Qwen-2.5 series, originally licensed under the Apache 2.0 License.

SpatialLM1.0 are built upon the SceneScript point cloud encoder, licensed under the CC-BY-NC-4.0 License. TorchSparse, utilized in this project, is licensed under the MIT License.

SpatialLM1.1 are built upon Sonata point cloud encoder, model weight is licensed under the CC-BY-NC-4.0 License. Code built on Pointcept is licensed under the Apache 2.0 License.

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{SpatialLM,
  title     = {SpatialLM: Training Large Language Models for Structured Indoor Modeling},
  author    = {Mao, Yongsen and Zhong, Junhao and Fang, Chuan and Zheng, Jia and Tang, Rui and Zhu, Hao and Tan, Ping and Zhou, Zihan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```

## Acknowledgements

We would like to thank the following projects that made this work possible:

[Llama3.2](https://github.com/meta-llama) | [Qwen2.5](https://github.com/QwenLM/Qwen2.5) | [Transformers](https://github.com/huggingface/transformers) | [SceneScript](https://github.com/facebookresearch/scenescript) | [TorchSparse](https://github.com/mit-han-lab/torchsparse) | [Sonata](https://xywu.me/sonata/) | [Pointcept](https://github.com/Pointcept/Pointcept)
