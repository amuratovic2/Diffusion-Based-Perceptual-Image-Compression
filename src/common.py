from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: str | Path) -> Iterable[Path]:
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_stem(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parts = rel.with_suffix("").parts
    return "__".join(parts)


def write_rows_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_bitrates(path: str | Path | None) -> dict[str, float]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row.get("filename")
            bpp = row.get("bpp")
            if name and bpp not in (None, ""):
                out[name] = float(bpp)
    return out


def file_bpp(path: str | Path, width: int, height: int) -> float:
    return os.path.getsize(path) * 8.0 / float(width * height)

