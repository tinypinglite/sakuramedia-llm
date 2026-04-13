from __future__ import annotations

import threading
import time
from pathlib import Path

from loguru import logger

from app.asr.audio import normalize_audio
from app.asr.engine import AsrEngine
from app.asr.schemas import DeviceChoice
from app.asr.service import TaskService
from app.asr.settings import Settings


class TaskWorker:
    def __init__(self, *, settings: Settings, service: TaskService, engine: AsrEngine):
        self.settings = settings
        self.service = service
        self.engine = engine
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._current_task_id: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run_forever, name="asr-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def snapshot(self) -> dict:
        return {
            "worker_enabled": self.settings.worker_enabled,
            "worker_alive": self.is_alive(),
            "current_task_id": self._current_task_id,
        }

    def run_forever(self) -> None:
        logger.info("ASR worker started")
        while not self._stop_event.is_set():
            handled = self.process_once()
            if not handled:
                time.sleep(self.settings.worker_poll_interval)
        logger.info("ASR worker stopped")

    def process_once(self) -> bool:
        task = self.service.claim_next_task()
        if task is None:
            return False

        task_logger = logger.bind(task_id=task.id)
        self._current_task_id = task.id
        upload_path = Path(task.upload_path)
        normalized_audio_path = upload_path.parent / "normalized.wav"
        try:
            task_logger.info("Processing task")
            normalize_audio(
                ffmpeg_binary=self.settings.ffmpeg_binary,
                input_path=upload_path,
                output_path=normalized_audio_path,
            )
            task_logger.info("Audio normalized")

            result = self.engine.transcribe(
                normalized_audio_path,
                model_size=task.model_size,
                device=DeviceChoice(task.requested_device),
                compute_type=task.compute_type,
                language=task.language,
                progress_callback=lambda progress, _current, duration: self.service.update_progress(
                    task.id,
                    progress=progress,
                    duration_seconds=duration,
                ),
            )
            self.service.mark_succeeded(task.id, normalized_audio_path=normalized_audio_path, result=result)
        except Exception as exc:  # pragma: no cover - exercised in tests
            self.service.mark_failed(task.id, str(exc))
        finally:
            self.service.cleanup_audio_files(task.id, upload_path, normalized_audio_path)
            self._current_task_id = None
        return True
