# ------------------------------------------------------------------------------
# Base: CUDA & Ubuntu
# ------------------------------------------------------------------------------
ARG CUDA_VERSION=11.8.0
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=20.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# ------------------------------------------------------------------------------
# Set global environment variables
# ------------------------------------------------------------------------------
ENV LANG C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"
    
    
# ------------------------------------------------------------------------------
# Install OS-level tools and dependencies 
# ------------------------------------------------------------------------------
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-get update -q && \
    DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends \
        bzip2 ca-certificates git libglib2.0-0 libsm6 libxext6 libxrender1 \
        mercurial openssh-client procps subversion wget curl vim unzip unrar \
        build-essential software-properties-common libgl1 \
        libboost-dev libopenexr-dev libeigen3-dev xvfb libgl1-mesa-glx \
        libglib2.0-0 ffmpeg openmpi-bin openmpi-common libopenmpi-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# Install Miniconda 
# ------------------------------------------------------------------------------
ARG CONDA_VERSION=py38_4.10.3
ARG PYTHON_VERSION=3.8

RUN set -x && \
    UNAME_M="$(uname -m)" && \
    if [ "${UNAME_M}" = "x86_64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh"; \
    fi && \
    wget "${MINICONDA_URL}" -O miniconda.sh -q && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda install python=${PYTHON_VERSION} -y && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    conda clean -afy

# ------------------------------------------------------------------------------
# Install base Python packages
# ------------------------------------------------------------------------------
RUN pip install \
    pyyaml h5py numpy scipy scikit-learn pandas pillow scikit-image \
    matplotlib seaborn jupyterlab jupyter

# ------------------------------------------------------------------------------
# Install PyTorch and CUDA-compiled packages 
# ------------------------------------------------------------------------------
ARG TORCH_VERSION=2.0.1
ARG TORCHVISION_VERSION=0.15.2
ARG TORCHSCATTER_VERSION=2.0.9
ARG SPCONV_VERSION=2.3.6
RUN export CU_VERSION=118 && \
    pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
        --index-url https://download.pytorch.org/whl/cu${CU_VERSION} && \
    pip install tensorboard \
        spconv-cu${CU_VERSION}==${SPCONV_VERSION} \
        torch-scatter==${TORCHSCATTER_VERSION} -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CU_VERSION}.html

# ------------------------------------------------------------------------------
# Install deep learning toolkits
# ------------------------------------------------------------------------------
ARG MMCV_VERSION=2.0.0
ARG MMDET_VERSION=3.00.0
ARG MMSEG_VERSION=1.2.0
RUN export CU_VERSION=118 && export TORCH_VER_SHORT=2.0 && \
    pip install mmcv==${MMCV_VERSION} \
        -f https://download.openmmlab.com/mmcv/dist/cu${CU_VERSION}/torch${TORCH_VER_SHORT}/index.html && \
    pip install mmsegmentation==${MMSEG_VERSION} mmdet==${MMDET_VERSION} ipdb

# ------------------------------------------------------------------------------
# Install 3D Vision and Dataset Tools 
# ------------------------------------------------------------------------------
RUN pip install \
    open3d==0.18.0 easydict==1.13 opencv-python==4.7.* \
    pyquaternion==0.9.9 SharedArray==3.2.4 kornia==0.6.* filterpy==1.4.5 motmetrics==1.4.0 \
    tensorflow==2.11.0 av2==0.2.1

# ------------------------------------------------------------------------------
# Extra utilities and dev tools 
# ------------------------------------------------------------------------------
RUN pip install \
    setuptools==59.5.0 yapf==0.40.1 protobuf==3.20.0 \
    einops==0.8.0 fvcore iopath==0.1.9 timm==0.6.13 typing-extensions==4.5.0 \
    pylint ipython==8.12 numpy==1.23.0 matplotlib==3.5.2 \
    llvmlite==0.32.1 numba==0.49.0 pandas==1.4.4 scikit-image==0.19.3 \
    setuptools==59.5.0 torchpack==0.3.1 wandb==0.18.0 tqdm transformers==4.46.3 \
    open_clip_torch==2.26.1 kaleido==0.2.1 pillow==10.4.0 tensorboardX==2.6.2.2

# ------------------------------------------------------------------------------
# ICP-Flow Dependencies 
# ------------------------------------------------------------------------------
RUN pip install \
    parmap torchist==0.2.0 hdbscan==0.8.38.post2 kiss-icp==0.4.0 \
    pytorch3d==0.7.4 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt201/download.html

# ------------------------------------------------------------------------------
# Create working directory 
# ------------------------------------------------------------------------------
RUN mkdir -p /OV-SCAN
WORKDIR /OV-SCAN

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
