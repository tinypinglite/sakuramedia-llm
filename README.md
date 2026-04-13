# sakuramedia-llm

轻量单机版异步 ASR 服务，使用 FastAPI + Peewee + SQLite + faster-whisper。

## 环境准备

```bash
conda env create -f environment.yml
conda activate sakuramedia-llm
poetry install
```

## 启动服务

```bash
export SAKURA_API_KEY="replace-with-a-secret"
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

`v1` 仅支持单实例、单 worker，请始终保持 `--workers 1`。

也可以写入项目根目录 `.env`：

```bash
SAKURA_API_KEY=replace-with-a-secret
```

`SAKURA_API_KEY` 为必填项，未配置时服务会在启动阶段直接失败。

## Docker 部署

项目提供两个镜像目标：

- `sakuramedia-llm:cpu`
- `sakuramedia-llm:cuda`

两者都保持当前服务形态不变，仍然使用单进程内 worker、SQLite 和本地文件系统。

### 构建镜像

构建 CPU 镜像：

```bash
docker build --target cpu -t sakuramedia-llm:cpu .
```

构建 CUDA 镜像：

```bash
docker build --target cuda -t sakuramedia-llm:cuda .
```

CUDA 镜像基于 `nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04`。宿主机需要提前安装兼容的 NVIDIA Driver 和 NVIDIA Container Toolkit。

### 运行镜像

推荐先准备持久化目录：

```bash
mkdir -p storage/tasks models
```

运行 CPU 镜像：

```bash
docker run --rm \
  -p 8000:8000 \
  -e SAKURA_API_KEY="replace-with-a-secret" \
  -v "$(pwd)/storage:/app/storage" \
  -v "$(pwd)/models:/app/models" \
  sakuramedia-llm:cpu
```

运行 CUDA 镜像：

```bash
docker run --rm \
  --gpus all \
  -p 8000:8000 \
  -e SAKURA_API_KEY="replace-with-a-secret" \
  -v "$(pwd)/storage:/app/storage" \
  -v "$(pwd)/models:/app/models" \
  sakuramedia-llm:cuda
```

容器内默认使用 `/app/storage` 保存 SQLite、上传文件和转录产物，使用 `/app/models` 作为本地模型目录和下载缓存目录。推荐将模型目录挂载为持久卷。

### 使用 Compose

仓库根目录提供了 `compose.yaml`。

启动 CPU 服务：

```bash
docker compose --profile cpu up --build asr-cpu
```

启动 CUDA 服务：

```bash
docker compose --profile cuda up --build asr-cuda
```

## 目录

- `app/main.py`: FastAPI 入口
- `app/asr/`: ASR 业务逻辑
- `storage/`: SQLite、上传文件和转录产物

## API

- `POST /api/v1/asr/tasks`
- `GET /api/v1/asr/tasks/{task_id}`
- `GET /api/v1/asr/tasks/{task_id}/result`
- `GET /healthz`

所有接口都必须携带请求头 `X-API-Key`，包括 `/healthz`。

示例：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/asr/tasks" \
  -H "X-API-Key: replace-with-a-secret" \
  -F "file=@sample.wav"
```

```bash
curl "http://127.0.0.1:8000/healthz" \
  -H "X-API-Key: replace-with-a-secret"
```
