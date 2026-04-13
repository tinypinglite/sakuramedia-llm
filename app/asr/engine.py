from __future__ import annotations

import importlib.util
import os
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from loguru import logger

from faster_whisper import WhisperModel

import ctranslate2

from app.asr.schemas import DeviceChoice

ProgressCallback = Callable[[float, float, float], None]


@dataclass(slots=True)
class SegmentResult:
    id: int
    start: float
    end: float
    text: str


@dataclass(slots=True)
class AsrResult:
    full_text: str
    srt_text: str
    segments: list[SegmentResult]
    language: str
    language_probability: float
    duration_seconds: float
    actual_device: str
    compute_type: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["segments"] = [asdict(segment) for segment in self.segments]
        return payload


class AsrEngine:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._models: dict[tuple[str, str, str], WhisperModel] = {}
        self._lock = threading.Lock()

    @staticmethod
    def prepare_cuda_runtime() -> None:
        if os.name != "posix":
            return

        package_names = ("nvidia.cublas.lib", "nvidia.cudnn.lib")
        lib_dirs: list[str] = []
        for pkg in package_names:
            try:
                spec = importlib.util.find_spec(pkg)
            except ModuleNotFoundError:
                spec = None
            if spec and spec.origin:
                lib_dir = str(Path(spec.origin).resolve().parent)
                if os.path.isdir(lib_dir):
                    lib_dirs.append(lib_dir)

        if not lib_dirs:
            return

        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        ld_parts = [part for part in current_ld_path.split(":") if part]
        missing = [item for item in lib_dirs if item not in ld_parts]
        if not missing:
            return

        merged: list[str] = []
        for value in [*lib_dirs, *ld_parts]:
            if value not in merged:
                merged.append(value)

        os.environ["LD_LIBRARY_PATH"] = ":".join(merged)
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)

    @staticmethod
    def cuda_available() -> bool:
        try:
            return bool(ctranslate2.get_supported_compute_types("cuda"))
        except Exception:
            return False

    @staticmethod
    def detect_compute_type(device: str, preferred: str | None) -> str:
        if preferred:
            return preferred

        if device != "cuda":
            return "int8"

        try:
            supported = ctranslate2.get_supported_compute_types("cuda")
        except Exception:
            return "int8_float16"

        for candidate in ("int8_float16", "float16", "int8", "float32"):
            if candidate in supported:
                return candidate

        return "float16"

    @staticmethod
    def resolve_device(requested_device: DeviceChoice) -> str:
        if requested_device == DeviceChoice.AUTO:
            raise RuntimeError("device=auto is not allowed")
        if requested_device == DeviceChoice.CPU:
            return "cpu"
        if requested_device == DeviceChoice.CUDA:
            if not AsrEngine.cuda_available():
                raise RuntimeError("CUDA requested but no CUDA runtime is available")
            return "cuda"
        raise RuntimeError(f"Unsupported device: {requested_device}")

    def _build_model(self, model_size: str, device: str, compute_type: str) -> WhisperModel:
        local_model_path = self.models_dir / f"faster-whisper-{model_size}"
        model_source = str(local_model_path) if local_model_path.exists() else model_size
        kwargs = {
            "device": device,
            "compute_type": compute_type,
        }
        if not local_model_path.exists():
            kwargs["download_root"] = str(self.models_dir)
        if device == "cpu":
            kwargs["cpu_threads"] = max(1, os.cpu_count() or 1)

        logger.info(
            "Loading Whisper model source={} device={} compute_type={}",
            model_source,
            device,
            compute_type,
        )
        return WhisperModel(model_source, **kwargs)

    def _get_model(self, model_size: str, device: str, compute_type: str) -> WhisperModel:
        key = (model_size, device, compute_type)
        with self._lock:
            model = self._models.get(key)
            if model is None:
                model = self._build_model(model_size, device, compute_type)
                self._models[key] = model
            return model

    @staticmethod
    def _format_srt_timestamp(seconds: float) -> str:
        total_ms = int(round(max(0.0, seconds) * 1000))
        hours, remainder = divmod(total_ms, 3600000)
        minutes, remainder = divmod(remainder, 60000)
        secs, milliseconds = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def transcribe(
        self,
        input_path: Path,
        *,
        model_size: str,
        device: DeviceChoice,
        compute_type: str | None,
        progress_callback: ProgressCallback | None = None,
    ) -> AsrResult:
        actual_device = self.resolve_device(device)
        resolved_compute_type = self.detect_compute_type(actual_device, compute_type)
        model = self._get_model(model_size, actual_device, resolved_compute_type)

        transcribe_kwargs = {
            "beam_size": 1,
            "best_of": 1,
            # 不传 language，让 Whisper 自行检测语言。
            "vad_filter": True,
            "vad_parameters": {
                "threshold": 0.3,
                "min_speech_duration_ms": 200,
                "min_silence_duration_ms": 600,
            },
            "condition_on_previous_text": False,
            "compression_ratio_threshold": 1.8,
        }
        segments, info = model.transcribe(str(input_path), **transcribe_kwargs)

        segment_results: list[SegmentResult] = []
        text_lines: list[str] = []
        srt_blocks: list[str] = []
        duration = float(getattr(info, "duration", 0.0) or 0.0)

        for idx, segment in enumerate(segments, start=1):
            text = segment.text.strip()
            result = SegmentResult(
                id=idx,
                start=float(segment.start),
                end=float(segment.end),
                text=text,
            )
            segment_results.append(result)
            text_lines.append(text)
            srt_blocks.append(
                f"{idx}\n"
                f"{self._format_srt_timestamp(result.start)} --> {self._format_srt_timestamp(result.end)}\n"
                f"{text}\n"
            )
            if progress_callback is not None:
                progress = min(1.0, result.end / duration) if duration > 0 else 0.0
                progress_callback(progress, result.end, duration)

        if progress_callback is not None:
            progress_callback(1.0, duration, duration)

        return AsrResult(
            full_text="\n".join(text_lines),
            srt_text="\n".join(srt_blocks).strip() + ("\n" if srt_blocks else ""),
            segments=segment_results,
            language=str(getattr(info, "language", "unknown")),
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            duration_seconds=duration,
            actual_device=actual_device,
            compute_type=resolved_compute_type,
        )
