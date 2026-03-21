from __future__ import annotations

import csv
import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import ExifTags, Image, UnidentifiedImageError
from pillow_heif import register_heif_opener
from tqdm import tqdm

import config_paths

missing = []

RAW_TAKEOUT_DIR = getattr(config_paths, "RAW_TAKEOUT_DIR", None)
MEDIA_ASSETS = getattr(config_paths, "MEDIA_ASSETS", None)
RAW_FILES_INDEX = getattr(config_paths, "RAW_FILES_INDEX", None)
ORPHAN_SIDECARS = getattr(config_paths, "ORPHAN_SIDECARS", None)
ARCHIVES_FOUND = getattr(config_paths, "ARCHIVES_FOUND", None)
AUDIT_REPORT = getattr(config_paths, "AUDIT_REPORT", None)

if RAW_TAKEOUT_DIR is None:
    missing.append("RAW_TAKEOUT_DIR")
if MEDIA_ASSETS is None:
    missing.append("MEDIA_ASSETS")
if RAW_FILES_INDEX is None:
    missing.append("RAW_FILES_INDEX")
if ORPHAN_SIDECARS is None:
    missing.append("ORPHAN_SIDECARS")
if ARCHIVES_FOUND is None:
    missing.append("ARCHIVES_FOUND")
if AUDIT_REPORT is None:
    missing.append("AUDIT_REPORT")

if missing:
    available = [attr for attr in dir(config_paths) if attr.isupper()]
    raise ImportError(
        f"config_paths.py не содержит: {', '.join(missing)}. Доступные переменные: {available}"
    )

register_heif_opener()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".3gp", ".m4v"}
SIDECAR_EXTENSIONS = {".json"}
ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

FACE_CASCADE_PATH = Path(__file__).with_name("haarcascade_frontalface_default.xml")
FACE_MIN_SIZE = 60
FACE_MIN_AREA = 12_000
FACE_MIN_COVERAGE = 0.03
HASH_CHUNK_SIZE = 4 * 1024 * 1024
ASSET_WORKERS = max(2, min(16, (os.cpu_count() or 8)))
THREAD_LOCAL = threading.local()


@dataclass
class RawFileRecord:
    file_path: str
    file_name: str
    extension: str
    file_size: int
    created_at_fs: float
    file_role: str
    inferred_asset_type: str
    album_path: str


@dataclass
class MediaAssetRecord:
    asset_id: str
    file_path: str
    primary_file_path: str
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
    sidecar_paths: str
    sidecar_count: int
    has_sidecar: bool
    content_type_file: Optional[str]


def sha256_file(path: Path, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def stable_asset_id(path: Path) -> str:
    normalized = str(path.resolve()).lower().encode("utf-8", errors="ignore")
    return hashlib.sha1(normalized).hexdigest()


def get_face_cascade() -> cv2.CascadeClassifier:
    cascade = getattr(THREAD_LOCAL, "face_cascade", None)
    if cascade is None:
        cascade = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
        THREAD_LOCAL.face_cascade = cascade
    return cascade


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


def estimate_face_coverage(rgb_image: Image.Image) -> float:
    arr = np.array(rgb_image)
    if arr.size == 0:
        return 0.0

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = get_face_cascade().detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(FACE_MIN_SIZE, FACE_MIN_SIZE),
    )
    if len(faces) == 0:
        return 0.0

    img_area = max(1, arr.shape[0] * arr.shape[1])
    total_face_area = 0
    for _, _, w, h in faces:
        area = int(w * h)
        if area >= FACE_MIN_AREA:
            total_face_area += area
    return float(total_face_area / img_area)


def classify_content_type(
    path: Path,
    width: Optional[int],
    height: Optional[int],
    face_coverage_estimate: float,
) -> Optional[str]:
    if width is None or height is None:
        return None

    ext = path.suffix.lower()
    is_tall_screen = width > 0 and height > 0 and (height / width) >= 1.8
    file_name_lower = path.name.lower()
    screenshot_like = any(token in file_name_lower for token in ("screenshot", "screen", "img_"))

    if (ext == ".png" or is_tall_screen) and face_coverage_estimate < 0.005:
        return "document"

    if face_coverage_estimate >= FACE_MIN_COVERAGE:
        return "people"

    if screenshot_like and ext == ".png" and face_coverage_estimate < 0.01:
        return "document"

    return "landscape"


def analyze_image(path: Path) -> tuple[Optional[str], Optional[int], Optional[int], Optional[str], Optional[str]]:
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            width, height = rgb.size
            exif_datetime = get_exif_datetime(img)

            try:
                phash = str(imagehash.phash(rgb))
            except Exception:
                phash = None

            face_coverage_estimate = estimate_face_coverage(rgb)
            content_type_file = classify_content_type(path, width, height, face_coverage_estimate)

            return phash, width, height, exif_datetime, content_type_file
    except (UnidentifiedImageError, OSError):
        return None, None, None, None, None


def infer_file_role(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "media", "photo"
    if ext in VIDEO_EXTENSIONS:
        return "media", "video"
    if ext in SIDECAR_EXTENSIONS:
        return "sidecar", "sidecar"
    if ext in ARCHIVE_EXTENSIONS:
        return "archive", "archive"
    return "other", "other"


def build_raw_file_record(path: Path, root: Path) -> RawFileRecord:
    stat = path.stat()
    file_role, inferred_asset_type = infer_file_role(path)
    try:
        album_path = str(path.parent.relative_to(root))
    except ValueError:
        album_path = str(path.parent)

    return RawFileRecord(
        file_path=str(path),
        file_name=path.name,
        extension=path.suffix.lower(),
        file_size=stat.st_size,
        created_at_fs=stat.st_mtime,
        file_role=file_role,
        inferred_asset_type=inferred_asset_type,
        album_path=album_path,
    )


def build_media_asset(path: Path, root: Path) -> MediaAssetRecord:
    ext = path.suffix.lower()
    is_image = ext in IMAGE_EXTENSIONS
    is_video = ext in VIDEO_EXTENSIONS

    stat = path.stat()
    sha256 = sha256_file(path)

    phash = None
    width = None
    height = None
    exif_datetime = None
    content_type_file = None

    if is_image:
        phash, width, height, exif_datetime, content_type_file = analyze_image(path)

    json_path = find_sidecar_json(path)
    json_datetime = None
    sidecar_paths: list[str] = []
    if json_path:
        meta = load_json_metadata(json_path)
        json_datetime = extract_json_datetime(meta)
        sidecar_paths.append(str(json_path))

    mime_type = "image" if is_image else "video" if is_video else "other"

    try:
        album_path = str(path.parent.relative_to(root))
    except ValueError:
        album_path = str(path.parent)

    return MediaAssetRecord(
        asset_id=stable_asset_id(path),
        file_path=str(path),
        primary_file_path=str(path),
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
        sidecar_paths=json.dumps(sidecar_paths, ensure_ascii=False),
        sidecar_count=len(sidecar_paths),
        has_sidecar=bool(sidecar_paths),
        content_type_file=content_type_file,
    )


def validate_paths() -> tuple[Path, dict[str, Path]]:
    root = Path(RAW_TAKEOUT_DIR)
    outputs = {
        "media_assets": Path(MEDIA_ASSETS),
        "raw_files": Path(RAW_FILES_INDEX),
        "orphan_sidecars": Path(ORPHAN_SIDECARS),
        "archives_found": Path(ARCHIVES_FOUND),
        "audit_report": Path(AUDIT_REPORT),
    }

    if not root.exists():
        raise FileNotFoundError(f"Не найден RAW_TAKEOUT_DIR: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"RAW_TAKEOUT_DIR не является папкой: {root}")

    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    return root.resolve(), {key: path.resolve() for key, path in outputs.items()}


def collect_all_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


def main() -> int:
    root, outputs = validate_paths()
    all_files = collect_all_files(root)

    print(f"takeout_dir = {root}")
    print(f"raw_files_csv = {outputs['raw_files']}")
    print(f"media_assets_csv = {outputs['media_assets']}")
    print(f"found_files = {len(all_files)}")
    print(f"asset_workers = {ASSET_WORKERS}")

    raw_records = []
    media_paths: list[Path] = []
    archive_rows: list[dict[str, object]] = []

    for path in tqdm(all_files, desc="Аудит входа"):
        raw_record = build_raw_file_record(path, root)
        raw_records.append(asdict(raw_record))

        if raw_record.file_role == "media":
            media_paths.append(path)
        elif raw_record.file_role == "archive":
            archive_rows.append(asdict(raw_record))

    raw_df = pd.DataFrame(raw_records).sort_values(["file_role", "album_path", "file_name", "file_path"])
    raw_df.to_csv(outputs["raw_files"], index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8-sig")

    asset_records = []
    linked_sidecars: set[str] = set()
    with ThreadPoolExecutor(max_workers=ASSET_WORKERS) as executor:
        for asset in tqdm(
            executor.map(lambda path: build_media_asset(path, root), media_paths),
            total=len(media_paths),
            desc="Сборка asset",
        ):
            asset_records.append(asdict(asset))
            if asset.json_path:
                linked_sidecars.add(asset.json_path)

    assets_df = pd.DataFrame(asset_records).sort_values(["album_path", "file_name", "file_path"])
    assets_df.to_csv(outputs["media_assets"], index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8-sig")

    orphan_sidecars_df = raw_df[
        (raw_df["file_role"] == "sidecar")
        & (~raw_df["file_path"].isin(linked_sidecars))
    ].copy()
    orphan_sidecars_df.to_csv(outputs["orphan_sidecars"], index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8-sig")

    archives_df = pd.DataFrame(archive_rows)
    if len(archives_df) == 0:
        archives_df = pd.DataFrame(columns=list(RawFileRecord.__annotations__.keys()))
    archives_df.to_csv(outputs["archives_found"], index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8-sig")

    audit_rows = [
        {"metric": "raw_files_total", "value": len(raw_df)},
        {"metric": "media_files_total", "value": int((raw_df["file_role"] == "media").sum())},
        {"metric": "sidecar_files_total", "value": int((raw_df["file_role"] == "sidecar").sum())},
        {"metric": "archive_files_total", "value": int((raw_df["file_role"] == "archive").sum())},
        {"metric": "other_files_total", "value": int((raw_df["file_role"] == "other").sum())},
        {"metric": "media_assets_total", "value": len(assets_df)},
        {"metric": "assets_with_sidecars", "value": int(assets_df["has_sidecar"].sum()) if len(assets_df) else 0},
        {"metric": "orphan_sidecars_total", "value": len(orphan_sidecars_df)},
    ]
    pd.DataFrame(audit_rows).to_csv(outputs["audit_report"], index=False, encoding="utf-8-sig")

    print(f"saved_raw_files_to = {outputs['raw_files']}")
    print(f"saved_media_assets_to = {outputs['media_assets']}")
    print(f"saved_orphan_sidecars_to = {outputs['orphan_sidecars']}")
    print(f"saved_archives_to = {outputs['archives_found']}")
    print(f"saved_audit_report_to = {outputs['audit_report']}")
    print(f"media_assets = {len(assets_df)}")
    print(f"orphan_sidecars = {len(orphan_sidecars_df)}")
    print(f"archives_found = {len(archives_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
