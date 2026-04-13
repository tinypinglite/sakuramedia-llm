from __future__ import annotations

from datetime import datetime

from peewee import CharField, DateTimeField, FloatField, Model, TextField

from app.asr.db import database_proxy


class BaseModel(Model):
    class Meta:
        database = database_proxy


class AsrTask(BaseModel):
    id = CharField(primary_key=True, max_length=64)
    status = CharField(max_length=32, index=True)
    original_filename = CharField(max_length=512)
    upload_path = TextField()
    normalized_audio_path = TextField(null=True)
    text_path = TextField(null=True)
    srt_path = TextField(null=True)
    result_json_path = TextField(null=True)
    model_size = CharField(max_length=128)
    requested_device = CharField(max_length=32)
    actual_device = CharField(max_length=32, null=True)
    compute_type = CharField(max_length=64, null=True)
    language = CharField(max_length=32)
    progress = FloatField(default=0.0)
    duration_seconds = FloatField(null=True)
    error_message = TextField(null=True)
    created_at = DateTimeField(default=datetime.utcnow)
    started_at = DateTimeField(null=True)
    finished_at = DateTimeField(null=True)
    updated_at = DateTimeField(default=datetime.utcnow)

    class Meta:
        table_name = "asr_tasks"
