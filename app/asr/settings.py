from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.asr.schemas import DeviceChoice, RuntimeDevicePolicy


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXED_MODEL_SIZE = "large-v3"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SAKURA_", env_file=".env", extra="ignore")

    app_name: str = "sakuramedia-llm"
    api_key: str
    storage_dir: Path = Field(default=REPO_ROOT / "storage")
    database_path: Path = Field(default=REPO_ROOT / "storage" / "asr.sqlite3")
    tasks_dir: Path = Field(default=REPO_ROOT / "storage" / "tasks")
    models_dir: Path = Field(default=REPO_ROOT / "models")
    ffmpeg_binary: str = "ffmpeg"
    runtime_device_policy: RuntimeDevicePolicy
    worker_poll_interval: float = 1.0
    worker_enabled: bool = True

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: str) -> str:
        # 静态密钥不能为空，避免服务在无保护状态下启动。
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("api_key must not be empty")
        return normalized_value

    def runtime_device_choice(self) -> DeviceChoice:
        return DeviceChoice(self.runtime_device_policy.value)

    def ensure_storage_dirs(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_storage_dirs()
    return settings
