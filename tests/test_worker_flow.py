from __future__ import annotations

from io import BytesIO
from pathlib import Path
import shutil

from fastapi import UploadFile

from app.asr.db import initialize_database
from app.asr.engine import AsrResult, SegmentResult
from app.asr.schemas import DeviceChoice, RuntimeDevicePolicy, TaskStatus
from app.asr.service import TaskService
from app.asr.worker import TaskWorker


class FakeEngine:
    def __init__(self):
        self.called_devices: list[DeviceChoice] = []

    def transcribe(self, input_path: Path, *, model_size, device, compute_type, progress_callback=None):
        del input_path, model_size, compute_type
        self.called_devices.append(device)
        if progress_callback:
            progress_callback(0.5, 5.0, 10.0)
            progress_callback(1.0, 10.0, 10.0)
        return AsrResult(
            full_text="line 1\nline 2",
            srt_text="1\n00:00:00,000 --> 00:00:05,000\nline 1\n\n2\n00:00:05,000 --> 00:00:10,000\nline 2\n",
            segments=[
                SegmentResult(id=1, start=0.0, end=5.0, text="line 1"),
                SegmentResult(id=2, start=5.0, end=10.0, text="line 2"),
            ],
            language="ja",
            language_probability=0.99,
            duration_seconds=10.0,
            actual_device="cpu",
            compute_type="int8",
        )


def test_worker_processes_task_successfully_with_cpu_policy(test_settings, monkeypatch):
    initialize_database(test_settings.database_path)
    service = TaskService(test_settings)
    service.create_tables()

    monkeypatch.setattr(
        "app.asr.worker.normalize_audio",
        lambda *, ffmpeg_binary, input_path, output_path: shutil.copyfile(input_path, output_path),
    )

    upload = UploadFile(filename="sample.wav", file=BytesIO(b"fake-audio"))
    task = service.create_task(upload=upload, compute_type=None)
    engine = FakeEngine()
    worker = TaskWorker(settings=test_settings, service=service, engine=engine)

    assert worker.process_once() is True

    stored = service.get_task(task.id)
    assert stored is not None
    assert stored.requested_device == RuntimeDevicePolicy.CPU.value
    assert stored.status == TaskStatus.SUCCEEDED.value
    assert stored.actual_device == "cpu"
    assert stored.compute_type == "int8"
    assert stored.progress == 1.0
    assert engine.called_devices == [DeviceChoice.CPU]
    assert not Path(stored.upload_path).exists()
    assert stored.normalized_audio_path is not None
    assert not Path(stored.normalized_audio_path).exists()
    assert Path(stored.result_json_path).exists()
    payload = service.load_result(task.id)
    assert payload["full_text"] == "line 1\nline 2"
    assert payload["segments"][0]["text"] == "line 1"


def test_worker_uses_cuda_policy_for_engine_device(test_settings, monkeypatch):
    test_settings.runtime_device_policy = RuntimeDevicePolicy.CUDA
    initialize_database(test_settings.database_path)
    service = TaskService(test_settings)
    service.create_tables()

    monkeypatch.setattr(
        "app.asr.worker.normalize_audio",
        lambda *, ffmpeg_binary, input_path, output_path: shutil.copyfile(input_path, output_path),
    )

    upload = UploadFile(filename="sample.wav", file=BytesIO(b"fake-audio"))
    task = service.create_task(upload=upload, compute_type=None)
    engine = FakeEngine()
    worker = TaskWorker(settings=test_settings, service=service, engine=engine)

    assert worker.process_once() is True

    stored = service.get_task(task.id)
    assert stored is not None
    assert stored.requested_device == RuntimeDevicePolicy.CUDA.value
    assert engine.called_devices == [DeviceChoice.CUDA]


def test_worker_cleans_up_audio_files_when_transcription_fails(test_settings, monkeypatch):
    initialize_database(test_settings.database_path)
    service = TaskService(test_settings)
    service.create_tables()

    monkeypatch.setattr(
        "app.asr.worker.normalize_audio",
        lambda *, ffmpeg_binary, input_path, output_path: shutil.copyfile(input_path, output_path),
    )

    class FailingEngine:
        def transcribe(self, input_path: Path, *, model_size, device, compute_type, progress_callback=None):
            del input_path, model_size, device, compute_type, progress_callback
            raise RuntimeError("transcribe failed")

    upload = UploadFile(filename="sample.wav", file=BytesIO(b"fake-audio"))
    task = service.create_task(upload=upload, compute_type=None)
    worker = TaskWorker(settings=test_settings, service=service, engine=FailingEngine())

    assert worker.process_once() is True

    stored = service.get_task(task.id)
    assert stored is not None
    assert stored.status == TaskStatus.FAILED.value
    assert not Path(stored.upload_path).exists()
    assert not Path(stored.upload_path).parent.joinpath("normalized.wav").exists()
