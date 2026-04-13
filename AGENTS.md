# AGENTS.md

这是一个为 SakuraMedia 项目配套的轻量异步 ASR 服务。

## 项目目标

- 通过 API 接收音频上传
- 将任务写入 SQLite 队列
- 在单个进程内后台异步执行转录
- 通过轮询接口返回结果

当前范围只包括音频转录。除非明确提出，不要把剧情总结、人声分离等能力加入主链路。

## 技术栈

- Python `3.11`
- `conda` 环境：`sakuramedia-llm`
- 依赖管理：`poetry`
- Web API：`FastAPI`
- ORM：`Peewee`
- 数据库：`SQLite`
- ASR：`faster-whisper` + `ctranslate2`
- 日志：`loguru`

## 环境规则

- 始终优先使用 conda 环境：

```bash
conda activate sakuramedia-llm
```

- 本项目不使用本地 `.venv`。
- Poetry 安装在 conda 环境中，默认应直接使用该解释器。
- 如果不想先激活 shell，可以这样执行一次性命令：

```bash
conda run -n sakuramedia-llm poetry run <command>
```

## 常用命令

- 安装依赖：

```bash
conda activate sakuramedia-llm
poetry install
```

- 运行测试：

```bash
poetry run pytest -q
```

- 启动 API：

```bash
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

## 运行约束

- 必须保持 `uvicorn --workers 1`。
- 当前 worker 是进程内单实例，底层使用 SQLite；多 worker 会带来重复消费和数据库竞争问题。
- `ffmpeg` 必须在 conda 环境中可用。
- 存储只使用本地文件系统。

## 架构说明

- [app/main.py]：FastAPI 应用工厂与生命周期管理
- `app/asr/api.py`：上传、状态、结果、健康检查接口
- `app/asr/service.py`：任务创建、状态流转、结果落盘
- `app/asr/worker.py`：单后台 worker 循环
- `app/asr/engine.py`：Whisper 模型加载、设备选择、转录执行
- `app/asr/audio.py`：音频校验与 `ffmpeg` 预处理
- `app/asr/models.py`：Peewee 数据模型
- `app/asr/settings.py`：应用配置与存储路径
- `storage/`：SQLite 文件和任务产物

## 行为约束

- 保持当前异步任务主链路：`upload -> queued -> processing -> succeeded/failed`。
- 上传音频后，必须先通过 `ffmpeg` 统一转成 `16k mono wav`，再交给转录引擎。
- 保持设备策略不变：
  - `auto`：优先 `cuda`，否则回退 `cpu`
  - `cuda`：不可用时直接失败
  - `cpu`：强制使用 CPU
- Whisper 模型按 `(model_size, device, compute_type)` 做缓存复用。
- 除非明确要求，不要随意改变 API 请求和响应结构。

## 测试要求

- 修改 ASR 相关逻辑时，要同步补充或更新 API、worker 流程、设备选择相关测试。
- 自动化测试优先使用 fake/mock 转录，不依赖真实 Whisper 模型下载。
- 单元测试不应要求联网拉模型。

## 改动原则

- 优先做小而直接的改动，保持当前服务形态稳定。
- 除非明确要求，不要提前引入独立队列系统、对象存储、鉴权、Webhook 等能力。
- 不要为了未来的剧情总结功能做过度抽象。
- 如果修改了请求字段、响应字段或运行方式，要同步更新测试和 README。
