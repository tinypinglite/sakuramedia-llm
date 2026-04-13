from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.asr.settings import Settings, get_settings
from app.main import create_app


def test_settings_reject_blank_api_key():
    with pytest.raises(ValidationError, match="api_key must not be empty"):
        Settings(api_key="   ")


def test_create_app_fails_when_api_key_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("SAKURA_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()

    app = create_app()

    try:
        with pytest.raises(ValidationError):
            with TestClient(app):
                pass
    finally:
        get_settings.cache_clear()
