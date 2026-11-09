# CUDA-enabled PyTorch base image; on a machine without a GPU it runs on the CPU.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
# build-essential: pybullet has no prebuilt wheel for every Python version
# xvfb + mesa: lets the PyBullet GUI / OpenGL renderer run inside the container
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git xvfb libgl1 libglib2.0-0 libegl1 libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
