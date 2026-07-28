from __future__ import annotations

from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def discover_images(path: str | Path, *, recursive: bool = False) -> list[Path]:
    """Return deterministic image paths for one file or directory."""

    source = Path(path).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source.suffix}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Image input does not exist: {source}")

    candidates: Iterable[Path] = source.rglob("*") if recursive else source.iterdir()
    return sorted(
        item for item in candidates
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )
