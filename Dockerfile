# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags
# FROM runpod/base:0.4.0-cuda11.8.0
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# The base image comes with many system dependencies pre-installed to help you get started quickly.
# Please refer to the base image's Dockerfile for more information before adding additional dependencies.
# IMPORTANT: The base image overrides the default huggingface cache location.


# --- Optional: System dependencies ---
# COPY builder/setup.sh /setup.sh
# RUN /bin/bash /setup.sh && \
#     rm /setup.sh


# Python dependencies
# COPY builder/requirements.txt /requirements.txt
# RUN --mount=type=cache,target=/root/.cache/pip \
#     python -m pip install  --upgrade pip && \
#     python -m pip install --default-timeout=100 --upgrade -r /requirements.txt --no-cache-dir && \
#     rm /requirements.txt

# NOTE: The base image comes with multiple Python versions pre-installed.
#       It is reccommended to specify the version of Python when running your code.
RUN python -m pip install  --upgrade pip
RUN pip install -q torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 torchtext==0.18.0 torchdata==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu121 \
    tqdm==4.66.5 numpy==1.26.3   xformers==0.0.27.post2  moviepy==1.0.3  sentencepiece==0.2.0 pillow==9.5.0 runpod
RUN pip install git+https://github.com/huggingface/diffusers
RUN pip install --upgrade transformers accelerate diffusers imageio-ffmpeg imageio
RUN pip install pillow sentencepiece opencv-python runpod==1.6.0
# transformers>=4.46.2
# accelerate>=1.1.1
# imageio-ffmpeg>=0.5.1
# Add src files (Worker Template)
ADD src .

RUN --mount=type=cache,target=/root/.cache/pip python /handler.py --default-timeout=100

CMD python -u /handler.py
