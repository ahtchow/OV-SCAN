# 📦 OV-SCAN Installation Guide

Follow the instructions below to install the requirements to build and run the setup for OV-SCAN.

## 🖥️ Step 1: Validate NVIDIA Driver Installation

Make sure your system has a compatible NVIDIA driver and CUDA version installed.

### 1.1 Check GPU Status
Run the following command:
```bash
nvidia-smi
```

### 1.2 What to Look For
The output should show:
- **NVIDIA driver version**
- **CUDA version**
- A list of your installed GPUs
- GPU memory and utilization stats

### 1.2 Expected Output
✅  You should see an output similar to the following:

```
Sun Jul 20 20:49:32 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.120                Driver Version: 550.120        CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla V100-SXM2-32GB           Off |   00000000:15:00.0 Off |                    0 |
| N/A   32C    P0             42W /  300W |       4MiB /  32768MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  Tesla V100-SXM2-32GB           Off |   00000000:16:00.0 Off |                    0 |
| N/A   33C    P0             41W /  300W |       4MiB /  32768MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  Tesla V100-SXM2-32GB           Off |   00000000:3A:00.0 Off |                    0 |
| N/A   30C    P0             41W /  300W |       4MiB /  32768MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  Tesla V100-SXM2-32GB           Off |   00000000:3B:00.0 Off |                    0 |
| N/A   33C    P0             42W /  300W |       4MiB /  32768MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   4  Tesla V100-SXM2-32GB           Off |   00000000:89:00.0 Off |                    0 |
| N/A   30C    P0             42W /  300W |       4MiB /  32768MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   5  Tesla V100-SXM2-32GB           Off |   00000000:8A:00.0 Off |                    0 |
| N/A   32C    P0             43W /  300W |       4MiB /  32768MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   6  Tesla V100-SXM2-32GB           Off |   00000000:B2:00.0 Off |                    0 |
| N/A   29C    P0             42W /  300W |       4MiB /  32768MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   7  Tesla V100-SXM2-32GB           Off |   00000000:B3:00.0 Off |                    0 |
| N/A   30C    P0             40W /  300W |       4MiB /  32768MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## 🔧 Step 2: Install or Update NVIDIA Drivers (If Needed)

If `nvidia-smi` does not show valid information or your CUDA version is incompatible, you’ll need to install or update the drivers.

### 2.1 Identify the Correct Driver for Your GPU
1. Visit the official [NVIDIA Driver Downloads](https://www.nvidia.com/en-us/drivers/).
2. Enter your GPU model (e.g., **A100**, **RTX 3090**, **TITAN**, etc.).
3. Select your OS and download the latest **production branch** driver.

### 2.2 Install the Driver
Replace `<version>` with your driver version (e.g., `550`):

```bash
sudo apt-get update
sudo apt-get install nvidia-driver-<version>
```

**Example:**
```bash
sudo apt-get install nvidia-driver-550
```

> ⚠️ On some systems (e.g., Ubuntu 22.04+), you may also need to run:
```bash
sudo ubuntu-drivers autoinstall
```

### 2.3 Reboot and Recheck
```bash
sudo reboot
```

After rebooting, confirm everything is working:
```bash
nvidia-smi
```

## 🐳 Step 3: Install Docker and NVIDIA Container Toolkit

Docker is required to run OV-SCAN in a containerized GPU environment.

### 3.1 Install Docker
Follow the official guide:  
👉 [Install Docker on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

Then follow the **post-installation steps** to enable non-root usage:  
👉 [Post-install steps](https://docs.docker.com/engine/install/linux-postinstall/)

💡 **Note:**  
> After adding your user to the `docker` group, you **must restart** the machine—not just log out/log in—for changes to take effect. This is also necessary to avoid `nvidia-smi` issues within Docker.

### 3.2 Test Docker
After reboot:
```bash
docker ps
```

✅ If successful, it should return an empty list of containers (not a permission error).

---

### 3.3 Install NVIDIA Container Toolkit

To enable GPU access within containers:

Follow this official guide:  
👉 [NVIDIA Container Toolkit Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)

> This allows Docker containers to use `nvidia-smi`, access GPUs, and run CUDA-based workloads inside the container.

You can verify installation by running:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu20.04 nvidia-smi
```

## 🧰 Step 4: Install `make`

The `make` utility is used to handle docker container execution.

### 4.1 Install via apt
```bash
sudo apt-get update
sudo apt-get install build-essential
```

This package includes `make`, `gcc`, and other essential build tools.

### 4.2 Verify Installation
```bash
make --version
```

✅ You should see version output like:
```
Built for x86_64-pc-linux-gnu
Copyright (C) 1988-2020 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
```

---

🎉 **Congratulations! You're all set up for OV-SCAN!** 🚀  
