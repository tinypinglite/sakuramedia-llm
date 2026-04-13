from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from app.asr.api import router as asr_router
from app.asr.audio import ensure_ffmpeg_available
from app.asr.db import close_database, initialize_database
from app.asr.engine import AsrEngine
from app.asr.service import TaskService
from app.asr.settings import get_settings
from app.asr.worker import TaskWorker


def create_app(*, settings_override=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = settings_override or get_settings()
        settings.ensure_storage_dirs()
        AsrEngine.prepare_cuda_runtime()
        ensure_ffmpeg_available(settings.ffmpeg_binary)
        initialize_database(settings.database_path)

        engine = AsrEngine(settings.models_dir)
        service = TaskService(settings)
        service.create_tables()
        worker = TaskWorker(settings=settings, service=service, engine=engine)

        app.state.settings = settings
        app.state.asr_engine = engine
        app.state.task_service = service
        app.state.task_worker = worker

        if settings.worker_enabled:
            worker.start()

        logger.info("Application startup complete")
        try:
            yield
        finally:
            worker.stop()
            close_database()
            logger.info("Application shutdown complete")

    app = FastAPI(title="sakuramedia-llm", lifespan=lifespan)
    app.include_router(asr_router)
    return app


app = create_app()
