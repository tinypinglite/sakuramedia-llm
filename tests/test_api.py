from __future__ import annotations

from io import BytesIO
from pathlib import Path
import shutil

from fastapi import UploadFile
import pytest

from app.asr.engine import AsrResult, SegmentResult
from app.asr.schemas import TaskStatus


@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/api/v1/asr/tasks", "post"),
        ("/api/v1/asr/tasks/missing-task", "get"),
        ("/api/v1/asr/tasks/missing-task/result", "get"),
        ("/healthz", "get"),
    ],
)
def test_all_routes_require_api_key(client, path, method):
    request = getattr(client, method)
    kwargs = {"files": {"file": ("audio.wav", b"fake-audio", "audio/wav")}} if method == "post" else {}

    response = request(path, **kwargs)

    assert response.status_code == 401
    assert response.json() == {"detail": "invalid api key"}


@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/api/v1/asr/tasks", "post"),
        ("/api/v1/asr/tasks/missing-task", "get"),
        ("/api/v1/asr/tasks/missing-task/result", "get"),
        ("/healthz", "get"),
    ],
)
def test_all_routes_reject_wrong_api_key(client, path, method):
    request = getattr(client, method)
    kwargs = {
        "headers": {"X-API-Key": "wrong-key"},
        "files": {"file": ("audio.wav", b"fake-audio", "audio/wav")},
    } if method == "post" else {"headers": {"X-API-Key": "wrong-key"}}

    response = request(path, **kwargs)

    assert response.status_code == 401
    assert response.json() == {"detail": "invalid api key"}


def test_create_task_and_poll_status(client, auth_headers):
    response = client.post(
        "/api/v1/asr/tasks",
        headers=auth_headers,
        files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
    )
    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == TaskStatus.QUEUED.value
    task_id = payload["task_id"]

    status_response = client.get(f"/api/v1/asr/tasks/{task_id}", headers=auth_headers)
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["task_id"] == task_id
    assert status_payload["status"] == TaskStatus.QUEUED.value
    assert status_payload["model_size"] == "large-v3"
    assert status_payload["requested_device"] == "cpu"
    assert status_payload["language"] == "auto"
    assert status_payload["compute_type"] is None

    result_response = client.get(f"/api/v1/asr/tasks/{task_id}/result", headers=auth_headers)
    assert result_response.status_code == 409
    assert result_response.json()["detail"]["status"] == TaskStatus.QUEUED.value


def test_create_task_ignores_device_and_language_form_fields(client, auth_headers):
    response = client.post(
        "/api/v1/asr/tasks",
        headers=auth_headers,
        files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
        data={"device": "cuda", "language": "en"},
    )
    assert response.status_code == 202
    payload = response.json()

    status_response = client.get(f"/api/v1/asr/tasks/{payload['task_id']}", headers=auth_headers)
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["requested_device"] == "cpu"
    assert status_payload["language"] == "auto"
    assert status_payload["model_size"] == "large-v3"


def test_get_result_for_completed_task(client, auth_headers):
    service = client.app.state.task_service
    task = service.create_task(
        upload=UploadFile(filename="audio.wav", file=BytesIO(b"fake-audio")),
        compute_type="int8",
    )
    task_dir = Path(task.upload_path).parent
    normalized_audio = task_dir / "normalized.wav"
    normalized_audio.write_bytes(b"wav")
    service.mark_succeeded(
        task.id,
        normalized_audio_path=normalized_audio,
        result=AsrResult(
            full_text="done",
            srt_text="1\n00:00:00,000 --> 00:00:01,000\ndone\n",
            segments=[SegmentResult(id=1, start=0.0, end=1.0, text="done")],
            language="ja",
            language_probability=1.0,
            duration_seconds=1.0,
            actual_device="cpu",
            compute_type="int8",
        ),
    )

    response = client.get(f"/api/v1/asr/tasks/{task.id}/result", headers=auth_headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == TaskStatus.SUCCEEDED.value
    assert payload["full_text"] == "done"
    assert payload["segments"][0]["text"] == "done"


def test_healthz(client, auth_headers):
    response = client.get("/healthz", headers=auth_headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["app_name"] == "sakuramedia-llm"
    assert payload["database_ok"] is True
    assert payload["ffmpeg_ok"] is True
    assert payload["runtime_device_policy"] == "cpu"
    assert payload["model_size"] == "large-v3"


def test_healthz_reflects_dependency_probe_failures(client, monkeypatch, auth_headers):
    monkeypatch.setattr("app.asr.api.ensure_connection", lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    monkeypatch.setattr(shutil, "which", lambda _binary: None)

    response = client.get("/healthz", headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["database_ok"] is False
    assert payload["ffmpeg_ok"] is False
    assert payload["runtime_device_policy"] == "cpu"
