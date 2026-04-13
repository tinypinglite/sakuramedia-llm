from __future__ import annotations

import secrets
import shutil

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile, status

from app.asr.audio import AudioValidationError
from app.asr.db import ensure_connection
from app.asr.engine import AsrEngine
from app.asr.schemas import (
    DeviceChoice,
    HealthResponse,
    TaskCreateResponse,
    TaskResultResponse,
    TaskStatus,
    TaskStatusResponse,
)
from app.asr.service import TaskService
from app.asr.settings import Settings
from app.asr.worker import TaskWorker

def get_task_service(request: Request) -> TaskService:
    return request.app.state.task_service


def get_worker(request: Request) -> TaskWorker:
    return request.app.state.task_worker


def get_settings_dependency(request: Request) -> Settings:
    return request.app.state.settings


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings_dependency),
) -> None:
    # 所有 HTTP 接口都要求固定请求头，禁止匿名访问。
    if x_api_key is None or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid api key")


router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/api/v1/asr/tasks", response_model=TaskCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_task(
    file: UploadFile = File(...),
    device: DeviceChoice = Form(DeviceChoice.AUTO),
    language: str | None = Form(None),
    task_service: TaskService = Depends(get_task_service),
    settings: Settings = Depends(get_settings_dependency),
) -> TaskCreateResponse:
    try:
        task = task_service.create_task(
            upload=file,
            device=device,
            language=language or settings.default_language,
        )
    except AudioValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    finally:
        await file.close()

    return TaskCreateResponse(task_id=task.id, status=TaskStatus(task.status))


@router.get("/api/v1/asr/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str, task_service: TaskService = Depends(get_task_service)) -> TaskStatusResponse:
    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="task not found")
    return TaskStatusResponse(**task_service.serialize_task(task))


@router.get("/api/v1/asr/tasks/{task_id}/result", response_model=TaskResultResponse)
def get_task_result(task_id: str, task_service: TaskService = Depends(get_task_service)) -> TaskResultResponse:
    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="task not found")
    if task.status != TaskStatus.SUCCEEDED.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"task_id": task.id, "status": task.status, "error_message": task.error_message},
        )

    result = task_service.load_result(task_id)
    return TaskResultResponse(task_id=task.id, status=TaskStatus(task.status), **result)


@router.get("/healthz", response_model=HealthResponse)
def healthz(
    worker: TaskWorker = Depends(get_worker),
    settings: Settings = Depends(get_settings_dependency),
) -> HealthResponse:
    try:
        ensure_connection().execute_sql("SELECT 1")
        database_ok = True
    except Exception:
        database_ok = False

    return HealthResponse(
        app_name=settings.app_name,
        database_ok=database_ok,
        ffmpeg_ok=shutil.which(settings.ffmpeg_binary) is not None,
        worker_enabled=settings.worker_enabled,
        worker_alive=worker.is_alive(),
        current_task_id=worker.snapshot()["current_task_id"],
        cuda_available=AsrEngine.cuda_available(),
        default_model_size=settings.default_model_size,
        default_language=settings.default_language,
    )
