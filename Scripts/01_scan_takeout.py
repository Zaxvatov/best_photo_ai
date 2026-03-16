from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import imagehash
import pandas as pd
from PIL import ExifTags, Image, UnidentifiedImageError
from pillow_heif import register_heif_opener
from tqdm import tqdm

from config_paths import MEDIA_INDEX_CSV, TAKEOUT_DIR

register_heif_opener()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".3gp", ".m4v"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


@dataclass
class MediaRecord:
    file_path: str
    file_name: str
    extension: str
    file_size: int
    created_at_fs: float
    sha256: str
    phash: Optional[str]
    width: Optional[int]
    height: Optional[int]
    exif_datetime: Optional[str]
    json_datetime: Optional[str]
    json_path: Optional[str]
    album_path: str
    mime_type: str
    is_image: bool
    is_video: bool


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()



def get_exif_datetime(img: Image.Image) -> Optional[str]:
    try:
        exif = img.getexif()
        if not exif:
            return None

        exif_map = {}
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            exif_map[tag] = value

        for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
            if key in exif_map:
                return str(exif_map[key])
    except Exception:
        return None
    return None



def load_json_metadata(json_path: Path) -> Optional[dict]:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None



def extract_json_datetime(meta: dict) -> Optional[str]:
    if not meta:
        return None

    candidates = [
        ("photoTakenTime", "timestamp"),
        ("creationTime", "timestamp"),
        ("modificationTime", "timestamp"),
    ]

    for a, b in candidates:
        if isinstance(meta.get(a), dict) and b in meta[a]:
            return str(meta[a][b])

    return None



def find_sidecar_json(media_path: Path) -> Optional[Path]:
    candidates = [
        media_path.with_name(media_path.name + ".json"),
        media_path.with_suffix(media_path.suffix + ".json"),
        media_path.with_suffix(".json"),
        media_path.parent / f"{media_path.stem}.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None



def analyze_image(path: Path) -> tuple[Optional[str], Optional[int], Optional[int], Optional[str]]:
    try:
        with Image.open(path) as img:
            width, height = img.size
            exif_datetime = get_exif_datetime(img)

            try:
                phash = str(imagehash.phash(img))
            except Exception:
                phash = None

            return phash, width, height, exif_datetime
    except (UnidentifiedImageError, OSError):
        return None, None, None, None



def build_record(path: Path, root: Path) -> MediaRecord:
    ext = path.suffix.lower()
    is_image = ext in IMAGE_EXTENSIONS
    is_video = ext in VIDEO_EXTENSIONS

    stat = path.stat()
    sha256 = sha256_file(path)

    phash = None
    width = None
    height = None
    exif_datetime = None

    if is_image:
        phash, width, height, exif_datetime = analyze_image(path)

    json_path = find_sidecar_json(path)
    json_datetime = None
    if json_path:
        meta = load_json_metadata(json_path)
        json_datetime = extract_json_datetime(meta)

    mime_type = "image" if is_image else "video" if is_video else "other"

    try:
        album_path = str(path.parent.relative_to(root))
    except ValueError:
        album_path = str(path.parent)

    return MediaRecord(
        file_path=str(path),
        file_name=path.name,
        extension=ext,
        file_size=stat.st_size,
        created_at_fs=stat.st_mtime,
        sha256=sha256,
        phash=phash,
        width=width,
        height=height,
        exif_datetime=exif_datetime,
        json_datetime=json_datetime,
        json_path=str(json_path) if json_path else None,
        album_path=album_path,
        mime_type=mime_type,
        is_image=is_image,
        is_video=is_video,
    )



def validate_paths() -> tuple[Path, Path]:
    root = Path(TAKEOUT_DIR)
    out_csv = Path(MEDIA_INDEX_CSV)

    if not root.exists():
        raise FileNotFoundError(f"Не найден TAKEOUT_DIR: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"TAKEOUT_DIR не является папкой: {root}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    return root.resolve(), out_csv.resolve()



def collect_media_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS:
            files.append(p)
    return files



def main() -> int:
    root, csv_path = validate_paths()
    all_files = collect_media_files(root)

    print(f"takeout_dir = {root}")
    print(f"output_csv = {csv_path}")
    print(f"found_media_files = {len(all_files)}")

    records = []
    for path in tqdm(all_files, desc="Сканирование"):
        records.append(asdict(build_record(path, root)))

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8-sig")

    print(f"saved_to = {csv_path}")
    print(f"rows = {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
