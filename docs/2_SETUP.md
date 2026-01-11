# 🛠️ OV-SCAN Setup Guide

Quick start guide to set up the OV-SCAN environment.

## 🐳 Step 1: Build Docker Container

Build the OV-SCAN Docker image:

```bash
make build
```

**Environment includes:**
- CUDA 11.8.0 + cuDNN 8 on Ubuntu 20.04
- Python 3.8 (Miniconda)
- PyTorch 2.0.1 + torchvision 0.15.2
- OpenMMLab (mmcv, mmdetection, mmsegmentation)
- 3D Vision (Open3D, PyTorch3D)
- Deep Learning (Transformers, CLIP, timm)

**Build time:** ~10-20 minutes

## 🚀 Step 2: Run Container

Start the container with GPU access:

```bash
make exec
```

**Mounts:**
- `./repos` → `/OV-SCAN/repos`
- `./datasets` → `/OV-SCAN/datasets`
- `./setup.sh` → `/OV-SCAN/setup.sh`

**To open additional terminals** in the same container:
```bash
make join
```

## 📁 Step 3: Prepare Datasets

Create the dataset directory:

```bash
mkdir -p datasets/nuscenes
```

Download and extract [NuScenes dataset](https://www.nuscenes.org/nuscenes) to `datasets/nuscenes/`

**Expected structure:**
```
datasets/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/
└── v1.0-test/
```

## ⚙️ Step 4: Run Setup Script (Inside Container)

**Inside the container**, run the setup script to compile CUDA extensions:

```bash
bash /OV-SCAN/setup.sh
```

**What it does:**
- ✅ Compiles & installs **DCNv4** (CUDA extensions require GPU)
- ✅ Builds **Patchwork++** Python wrapper for ground segmentation
- ✅ Configures **NuScenes Devkit** PYTHONPATH
- ✅ Sets up **ImmortalTracker** for 3D object tracking
- ✅ Verifies all installations

**First run:** ~5-10 minutes (compiles CUDA code)

## 🧪 Step 5: Verify Installation

The setup script automatically verifies installations. You should see:

```
✓ pypatchworkpp imported successfully
✓ DCNv4 imported successfully
✓ nuscenes-devkit imported successfully
✓ ImmortalTracker imported successfully
```

---

## 🚀 Quick Start Summary

```bash
# 1. Build (one-time)
make build

# 2. Start container
make exec

# 3. Inside container: Run setup
bash /OV-SCAN/setup.sh

# 4. You're ready!
```

🎉 **All set! Start developing with OV-SCAN!**