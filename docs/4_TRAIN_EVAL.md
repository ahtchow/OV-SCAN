# 🚀 OV-SCAN Training & Evaluation Guide

This guide explains how to train and evaluate OV-SCAN models for open-vocabulary 3D object detection.

---

## 📋 Prerequisites

Before starting training, ensure you have:

- ✅ Completed [Environment Setup](2_ENV_SETUP.md)
- ✅ Completed [Data Setup](3_DATA_SETUP.md)
- ✅ NuScenes dataset properly configured
- ✅ All pre-computed files in place

---

## 🏋️ Training

### Training Workflow

**Recommended approach:** Train the LiDAR-only model first, then use it to initialize the multimodal model.

### Step 1: Train LiDAR-Only Model

Train the LiDAR-only 3D object detection model from scratch:

```bash
# Inside the Docker container
cd /OV-SCAN/OV-SCAN/tools

# Multi-GPU training (8 GPUs) - Recommended
scripts/dist_train.sh 8 \
    --cfg_file cfgs/nuscenes_models/ov_scan_lidar.yaml
```

**Output:** Checkpoints saved to `output/ov_scan_lidar/default/ckpt/`

### Step 2: Train Multimodal Model

After LiDAR training completes, train the multimodal model using the LiDAR checkpoint:

```bash
# Inside the Docker container
cd /OV-SCAN/OV-SCAN/tools

# Multi-GPU training (8 GPUs) - Recommended
scripts/dist_train.sh 8 \
    --cfg_file cfgs/nuscenes_models/ov_scan_multimodal.yaml \
    --pretrained_model ../pretrained/<path_to_ov_scan_lidar_model>.pth
```

**Output:** Checkpoints saved to `output/ov_scan_multimodal/default/ckpt/`

---

## 📊 Evaluation

### Evaluate Trained Model

Evaluate a trained model on the validation set:

```bash
# Inside the Docker container
cd /OV-SCAN/OV-SCAN/tools

# Evaluate LiDAR-only model (8 GPUs)
scripts/dist_test.sh 8 \
    --cfg_file cfgs/nuscenes_models/ov_scan_lidar.yaml \
    --ckpt ../pretrained/ov_scan_lidar.pth

# Evaluate multimodal model (8 GPUs)
scripts/dist_test.sh 8 \
    --cfg_file cfgs/nuscenes_models/ov_scan_multimodal.yaml \
    --ckpt ../pretrained/ov_scan_multimodal.pth
```
