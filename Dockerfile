# syntax=docker/dockerfile:1

FROM python:3.11-slim AS cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SAKURA_RUNTIME_DEVICE_POLICY=cpu

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU 镜像仅使用 requirements.txt 安装运行依赖。
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app ./app

RUN mkdir -p /app/storage/tasks /app/models

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS cuda

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SAKURA_RUNTIME_DEVICE_POLICY=cuda

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CUDA 镜像使用系统 Python 3.10 + pip，不再引入 conda/poetry。
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY app ./app

RUN mkdir -p /app/storage/tasks /app/models

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
