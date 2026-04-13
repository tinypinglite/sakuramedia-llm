from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class DeviceChoice(str, Enum):
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class RuntimeDevicePolicy(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"


class SegmentResponse(BaseModel):
    id: int
    start: float
    end: float
    text: str


class TaskCreateResponse(BaseModel):
    task_id: str
    status: TaskStatus


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    original_filename: str
    model_size: str
    requested_device: DeviceChoice
    actual_device: str | None = None
    compute_type: str | None = None
    language: str
    progress: float = Field(ge=0.0, le=1.0)
    duration_seconds: float | None = None
    error_message: str | None = None
    has_text: bool
    has_srt: bool
    has_result: bool
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime


class TaskResultResponse(BaseModel):
    task_id: str
    status: TaskStatus
    full_text: str
    srt_text: str
    segments: list[SegmentResponse]
    language: str
    duration_seconds: float
    actual_device: str
    compute_type: str


class HealthResponse(BaseModel):
    app_name: str
    database_ok: bool
    ffmpeg_ok: bool
    worker_enabled: bool
    worker_alive: bool
    current_task_id: str | None = None
    cuda_available: bool
    runtime_device_policy: RuntimeDevicePolicy
    model_size: str
