FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set envs for non-interactive install
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Create a dockeruser to match preprocessing.py expectations
RUN useradd -m dockeruser

# Install system dependencies and Python
RUN apt-get update --fix-missing && apt-get install -y \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    ca-certificates \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python packages
RUN pip install \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    torch==2.6.0 \
    torchvision==0.21.0 \
    monai==1.5.0 \
    simpleitk==2.3.1 \
    tqdm \
    einops \
    nibabel

# Set FSL install directory
ENV FSLDIR=/opt/fsl

# Remove any previous FSL install (optional, for clean build)
RUN rm -rf $FSLDIR

# Install FSL into $FSLDIR
RUN curl -Ls https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/getfsl.sh | sh -s

# Add FSL environment setup to all bash shells
RUN echo "source $FSLDIR/etc/fslconf/fsl.sh" >> /etc/bash.bashrc

# Add FSL binaries to PATH
ENV PATH="$FSLDIR/bin:$PATH"

# === Copy project files into container ===
WORKDIR /workspace

# Copy BM-MAE code
COPY BM-MAE /workspace/BM-MAE

# Copy checkpoint
COPY checkpoint /workspace/checkpoint

# Copy preprocessing & inference scripts
COPY scripts/preprocessing.py /workspace/preprocessing.py
COPY scripts/inference.py /workspace/inference.py
COPY run_inference.sh /workspace/run_inference.sh

# Make run_inference.sh executable
RUN chmod +x /workspace/run_inference.sh

# Set default entrypoint to bash (challenge will override with run_inference.sh)
ENTRYPOINT ["/bin/bash"]
