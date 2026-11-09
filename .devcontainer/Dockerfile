# PyTorch 2.3.1 with CUDA 12.1 and cuDNN 8
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Avoid interactive installation prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    openssh-client \
    ca-certificates \
    xvfb \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspaces/Learning2Rearrange

# Install Python dependencies
# Copy requirements separately to leverage Docker layer caching
COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install \
    --no-cache-dir \
    -r /tmp/requirements.txt

# Default command
CMD ["bash"]