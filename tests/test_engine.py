from __future__ import annotations

import pytest

from app.asr.engine import AsrEngine
from app.asr.schemas import DeviceChoice


class FakeCTranslate2:
    @staticmethod
    def get_supported_compute_types(device: str):
        if device == "cuda":
            return ["float16", "int8_float16"]
        return ["int8"]


class BrokenCTranslate2:
    @staticmethod
    def get_supported_compute_types(device: str):
        raise RuntimeError("boom")


def test_resolve_device_rejects_auto():
    with pytest.raises(RuntimeError, match="device=auto is not allowed"):
        AsrEngine.resolve_device(DeviceChoice.AUTO)


def test_resolve_device_cuda_requires_runtime(monkeypatch):
    monkeypatch.setattr(AsrEngine, "cuda_available", staticmethod(lambda: False))
    with pytest.raises(RuntimeError, match="CUDA requested"):
        AsrEngine.resolve_device(DeviceChoice.CUDA)


def test_detect_compute_type_prefers_gpu_mixed_precision(monkeypatch):
    monkeypatch.setattr("app.asr.engine.ctranslate2", FakeCTranslate2())
    assert AsrEngine.detect_compute_type("cuda", None) == "int8_float16"


def test_detect_compute_type_handles_cuda_probe_failure(monkeypatch):
    monkeypatch.setattr("app.asr.engine.ctranslate2", BrokenCTranslate2())
    assert AsrEngine.detect_compute_type("cuda", None) == "int8_float16"


def test_detect_compute_type_defaults_cpu_to_int8():
    assert AsrEngine.detect_compute_type("cpu", None) == "int8"


def test_build_model_uses_local_models_dir_when_present(monkeypatch, tmp_path):
    captured = {}

    def fake_whisper_model(source, **kwargs):
        captured["source"] = source
        captured["kwargs"] = kwargs
        return object()

    local_model_dir = tmp_path / "faster-whisper-large-v3"
    local_model_dir.mkdir()

    monkeypatch.setattr("app.asr.engine.WhisperModel", fake_whisper_model)

    engine = AsrEngine(tmp_path)
    engine._build_model("large-v3", "cpu", "int8")

    assert captured["source"] == str(local_model_dir)
    assert "download_root" not in captured["kwargs"]
    assert captured["kwargs"]["compute_type"] == "int8"
    assert captured["kwargs"]["device"] == "cpu"
    assert captured["kwargs"]["cpu_threads"] >= 1


def test_build_model_persists_downloads_under_models_dir(monkeypatch, tmp_path):
    captured = {}

    def fake_whisper_model(source, **kwargs):
        captured["source"] = source
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("app.asr.engine.WhisperModel", fake_whisper_model)

    engine = AsrEngine(tmp_path)
    engine._build_model("large-v3", "cuda", "float16")

    assert captured["source"] == "large-v3"
    assert captured["kwargs"]["device"] == "cuda"
    assert captured["kwargs"]["compute_type"] == "float16"
    assert captured["kwargs"]["download_root"] == str(tmp_path)
