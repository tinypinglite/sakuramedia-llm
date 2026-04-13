from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.asr.engine import AsrEngine
from app.asr.schemas import RuntimeDevicePolicy
from app.asr.settings import Settings, get_settings
from app.main import create_app


def test_settings_reject_blank_api_key():
    with pytest.raises(ValidationError, match="api_key must not be empty"):
        Settings(api_key="   ", runtime_device_policy="cpu")


def test_settings_require_runtime_device_policy():
    with pytest.raises(ValidationError, match="runtime_device_policy"):
        Settings(api_key="test-api-key")


def test_settings_reject_auto_runtime_device_policy():
    with pytest.raises(ValidationError, match="runtime_device_policy"):
        Settings(api_key="test-api-key", runtime_device_policy="auto")


def test_create_app_fails_when_api_key_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("SAKURA_API_KEY", raising=False)
    monkeypatch.setenv("SAKURA_RUNTIME_DEVICE_POLICY", "cpu")
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()

    app = create_app()

    try:
        with pytest.raises(ValidationError):
            with TestClient(app):
                pass
    finally:
        get_settings.cache_clear()


def test_create_app_fails_when_runtime_policy_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKURA_API_KEY", "test-api-key")
    monkeypatch.delenv("SAKURA_RUNTIME_DEVICE_POLICY", raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()

    app = create_app()

    try:
        with pytest.raises(ValidationError, match="runtime_device_policy"):
            with TestClient(app):
                pass
    finally:
        get_settings.cache_clear()


def test_create_app_fails_fast_when_runtime_policy_is_cuda_but_cuda_unavailable(monkeypatch, test_settings):
    test_settings.runtime_device_policy = RuntimeDevicePolicy.CUDA
    monkeypatch.setattr(AsrEngine, "cuda_available", staticmethod(lambda: False))

    app = create_app(settings_override=test_settings)
    with pytest.raises(RuntimeError, match="runtime_device_policy=cuda requires a working CUDA runtime"):
        with TestClient(app):
            pass


def test_create_app_starts_when_runtime_policy_is_cpu_even_if_no_cuda(monkeypatch, test_settings):
    test_settings.runtime_device_policy = RuntimeDevicePolicy.CPU
    monkeypatch.setattr(AsrEngine, "cuda_available", staticmethod(lambda: False))

    app = create_app(settings_override=test_settings)
    with TestClient(app):
        pass
