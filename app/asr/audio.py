from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".oga",
    ".ogg",
    ".wav",
    ".webm",
    ".wma",
}


class AudioValidationError(ValueError):
    pass


class AudioNormalizationError(RuntimeError):
    pass


def is_supported_audio_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def ensure_ffmpeg_available(binary: str) -> None:
    if shutil.which(binary) is None:
        raise RuntimeError(f"ffmpeg binary not found: {binary}")


def normalize_audio(*, ffmpeg_binary: str, input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_binary,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]

    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "unknown ffmpeg error"
        raise AudioNormalizationError(f"failed to normalize audio: {stderr}")
