# syntax=docker/dockerfile:1

ARG POETRY_VERSION=1.8.3

FROM python:3.11-slim AS cpu

ARG POETRY_VERSION

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock poetry.toml README.md ./
RUN poetry install --only main --no-root --no-ansi

COPY app ./app

RUN mkdir -p /app/storage/tasks /app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS cuda

ARG POETRY_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PATH=/opt/conda/envs/sakuramedia-llm/bin:/opt/conda/bin:$PATH

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends bzip2 ca-certificates curl ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py311_24.11.1-0-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm -f /tmp/miniconda.sh \
    && "${CONDA_DIR}/bin/conda" create -y -n sakuramedia-llm python=3.11 pip \
    && "${CONDA_DIR}/bin/conda" clean -afy

RUN pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock poetry.toml README.md ./
RUN poetry install --only main --no-root --no-ansi

COPY app ./app

RUN mkdir -p /app/storage/tasks /app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
