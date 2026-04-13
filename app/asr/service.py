from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from loguru import logger

from app.asr.audio import AudioValidationError, is_supported_audio_filename
from app.asr.db import ensure_connection, get_database
from app.asr.engine import AsrResult
from app.asr.models import AsrTask
from app.asr.schemas import DeviceChoice, TaskStatus
from app.asr.settings import Settings


class TaskService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def create_tables(self) -> None:
        db = ensure_connection()
        db.create_tables([AsrTask])

    def create_task(
        self,
        *,
        upload: UploadFile,
        device: DeviceChoice,
        language: str,
        model_size: str | None = None,
        compute_type: str | None = None,
    ) -> AsrTask:
        filename = upload.filename or "upload.bin"
        if not is_supported_audio_filename(filename):
            raise AudioValidationError(f"unsupported audio file: {filename}")

        task_id = uuid4().hex
        task_dir = self.settings.tasks_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        upload_suffix = Path(filename).suffix.lower() or ".bin"
        upload_path = task_dir / f"input{upload_suffix}"

        upload.file.seek(0)
        with upload_path.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)

        now = datetime.utcnow()
        ensure_connection()
        task = AsrTask.create(
            id=task_id,
            status=TaskStatus.QUEUED.value,
            original_filename=filename,
            upload_path=str(upload_path),
            model_size=model_size or self.settings.default_model_size,
            requested_device=device.value,
            compute_type=compute_type or None,
            language=language,
            created_at=now,
            updated_at=now,
        )
        logger.bind(task_id=task_id).info("Task queued for file {}", filename)
        return task

    def get_task(self, task_id: str) -> AsrTask | None:
        ensure_connection()
        return AsrTask.get_or_none(AsrTask.id == task_id)

    def claim_next_task(self) -> AsrTask | None:
        db = ensure_connection()
        with db.atomic():
            task = (
                AsrTask.select()
                .where(AsrTask.status == TaskStatus.QUEUED.value)
                .order_by(AsrTask.created_at)
                .first()
            )
            if task is None:
                return None

            updated = (
                AsrTask.update(
                    status=TaskStatus.PROCESSING.value,
                    started_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    progress=0.0,
                    error_message=None,
                )
                .where(
                    (AsrTask.id == task.id)
                    & (AsrTask.status == TaskStatus.QUEUED.value)
                )
                .execute()
            )
            if updated == 0:
                return None

        return self.get_task(task.id)

    def update_progress(self, task_id: str, *, progress: float, duration_seconds: float | None) -> None:
        ensure_connection()
        AsrTask.update(
            progress=max(0.0, min(1.0, progress)),
            duration_seconds=duration_seconds,
            updated_at=datetime.utcnow(),
        ).where(AsrTask.id == task_id).execute()

    def mark_failed(self, task_id: str, error_message: str) -> None:
        logger.bind(task_id=task_id).exception("Task failed: {}", error_message)
        ensure_connection()
        AsrTask.update(
            status=TaskStatus.FAILED.value,
            error_message=error_message,
            finished_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ).where(AsrTask.id == task_id).execute()

    def mark_succeeded(
        self,
        task_id: str,
        *,
        normalized_audio_path: Path,
        result: AsrResult,
    ) -> None:
        task_dir = normalized_audio_path.parent
        text_path = task_dir / "transcript.txt"
        srt_path = task_dir / "transcript.srt"
        result_json_path = task_dir / "result.json"

        text_path.write_text(result.full_text, encoding="utf-8")
        srt_path.write_text(result.srt_text, encoding="utf-8")
        result_json_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        ensure_connection()
        AsrTask.update(
            status=TaskStatus.SUCCEEDED.value,
            normalized_audio_path=str(normalized_audio_path),
            text_path=str(text_path),
            srt_path=str(srt_path),
            result_json_path=str(result_json_path),
            actual_device=result.actual_device,
            compute_type=result.compute_type,
            progress=1.0,
            duration_seconds=result.duration_seconds,
            finished_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            error_message=None,
        ).where(AsrTask.id == task_id).execute()
        logger.bind(task_id=task_id).info("Task completed successfully")

    def cleanup_audio_files(self, task_id: str, *paths: Path) -> None:
        task_logger = logger.bind(task_id=task_id)
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                task_logger.warning("Failed to remove audio file {}: {}", path, exc)

    def load_result(self, task_id: str) -> dict:
        task = self.get_task(task_id)
        if task is None:
            raise KeyError(task_id)
        if task.result_json_path is None:
            raise FileNotFoundError(task_id)
        return json.loads(Path(task.result_json_path).read_text(encoding="utf-8"))

    @staticmethod
    def serialize_task(task: AsrTask) -> dict:
        return {
            "task_id": task.id,
            "status": task.status,
            "original_filename": task.original_filename,
            "model_size": task.model_size,
            "requested_device": task.requested_device,
            "actual_device": task.actual_device,
            "compute_type": task.compute_type,
            "language": task.language,
            "progress": task.progress,
            "duration_seconds": task.duration_seconds,
            "error_message": task.error_message,
            "has_text": bool(task.text_path and Path(task.text_path).exists()),
            "has_srt": bool(task.srt_path and Path(task.srt_path).exists()),
            "has_result": bool(task.result_json_path and Path(task.result_json_path).exists()),
            "created_at": task.created_at,
            "started_at": task.started_at,
            "finished_at": task.finished_at,
            "updated_at": task.updated_at,
        }

    def reset(self) -> None:
        db = get_database()
        db.drop_tables([AsrTask])
        db.create_tables([AsrTask])
