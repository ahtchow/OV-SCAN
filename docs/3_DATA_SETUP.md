# 📦 OV-SCAN Data Setup Guide

This guide explains how to set up the training data files for OV-SCAN. Choose between using pre-computed files (recommended) or generating them from scratch.

---

## ⚡ Quick Setup (Recommended)

**Use pre-computed files to get started immediately** - Skip the time-consuming data generation process.

### Download Pre-computed Files

Download all required files from [Google Drive](https://drive.google.com/drive/folders/1I0EC9DYN7Op2rKJRZx1M-NPBPpfkQKME?usp=sharing):

| File | Description | Extract To |
|------|-------------|------------|
| `nuscenes_infos_10sweeps_train_ov_task.pkl` | Training dataset info | `data/nuscenes/v1.0-trainval/` |
| `nuscenes_dbinfos_10sweeps_withvelo_full_ov.pkl` | Database metadata | `data/nuscenes/v1.0-trainval/` |
| `gt_database_10sweeps_withvelo_full_ov.zip` | Ground truth object database | `data/nuscenes/v1.0-trainval/` |
| `train.zip` | Trained models & object embeddings | `data/nuscenes/v1.0-trainval/ov_alignment` |
| `pretrained.zip` | Pretrained backbone and model weights | `pretrained/` |

### Expected Directory Structure

Unzip and place the files to match the following structure:

```
OV-SCAN/
├── data/
│   └── nuscenes/
│       ├── maps/                          # NuScenes map files
│       ├── samples/                       # NuScenes sample data
│       ├── sweeps/                        # NuScenes sweep data
│       ├── v1.0-trainval/
│       │   ├── gt_database_10sweeps_withvelo_full_ov/    # ← Extract gt_database zip here
│       │   │   ├── <object_id_1>.bin      # Individual object point clouds
│       │   │   ├── <object_id_2>.bin
│       │   │   ├── <object_id_3>.bin
│       │   │   └── ...                    # More object files
│       │   ├── ov_alignment/             
│       │   │   └── DFN5B-CLIP-ViT-H-14/
│       │   │       └── object_crops/
│       │   │           └── train/         # ← Extract from train.zip
│       │   ├── nuscenes_dbinfos_10sweeps_withvelo_full_ov.pkl    # ← Copy here
│       │   ├── nuscenes_infos_10sweeps_train_full_ov_task.pkl    # ← Copy here
│       └── v1.0-test/
│
├── pretrained/                              # ← Extract pretrained.zip here
│   ├── flash_intern_image_s_1k_224.pth      # InternImage backbone weights
│   ├── ov_scan_lidar.pth                    # LiDAR-only trained model
│   └── ov_scan_multimodal.pth               # Multimodal trained model
│
└── OV-SCAN/
    └── tools/
        └── cfgs/
            └── dataset_configs/
                └── nuscenes_full_ov_dataset.yaml
```

---

## 🔧 Generate Files from Scratch

> **Coming Soon** - Instructions for generating each file individually from raw NuScenes data.

If you want to generate the training files yourself, detailed instructions will be added here for:

1. **Training Dataset Info** (`nuscenes_infos_10sweeps_train_ov_task.pkl`)
2. **Database Metadata** (`nuscenes_dbinfos_10sweeps_withvelo_full_ov.pkl`)
3. **Ground Truth Database** (`gt_database_10sweeps_withvelo_full_ov/`)
4. **Object Embeddings** (`ov_alignment/DFN5B-CLIP-ViT-H-14/object_crops/train/`)

For now, please use the [pre-computed files](#-quick-setup-recommended) above.

---