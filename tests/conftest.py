from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.asr.db import close_database
from app.asr.settings import Settings
from app.main import create_app


@pytest.fixture()
def test_settings(tmp_path: Path) -> Settings:
    settings = Settings(
        api_key="test-api-key",
        storage_dir=tmp_path / "storage",
        database_path=tmp_path / "storage" / "test.sqlite3",
        tasks_dir=tmp_path / "storage" / "tasks",
        models_dir=tmp_path / "models",
        worker_enabled=False,
        worker_poll_interval=0.01,
    )
    settings.ensure_storage_dirs()
    return settings


@pytest.fixture()
def client(test_settings: Settings):
    app = create_app(settings_override=test_settings)
    with TestClient(app) as test_client:
        yield test_client
    close_database()


@pytest.fixture()
def auth_headers(test_settings: Settings) -> dict[str, str]:
    return {"X-API-Key": test_settings.api_key}
