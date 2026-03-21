from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config_paths as cfg


REQUIRED_CONFIG_ATTRS = (
    "UNIQUE_MEDIA",
    "VIDEO_INDEX",
)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
    ".3gp",
    ".mts",
    ".m2ts",
    ".wmv",
}


def validate_config() -> None:
    missing = [name for name in REQUIRED_CONFIG_ATTRS if not hasattr(cfg, name)]
    if missing:
        available = sorted(name for name in dir(cfg) if name.isupper())
        raise ImportError(
            "config_paths.py не содержит: "
            + ", ".join(missing)
            + ". Доступные переменные: "
            + str(available)
        )


def resolve_io_paths(index_dir: Path | None) -> tuple[Path, Path]:
    validate_config()

    if index_dir is None:
        return Path(cfg.UNIQUE_MEDIA), Path(cfg.VIDEO_INDEX)

    return index_dir / Path(cfg.UNIQUE_MEDIA).name, index_dir / Path(cfg.VIDEO_INDEX).name


def is_video_row(row: pd.Series) -> bool:
    if bool(row.get("is_video", False)):
        return True

    mime_type = str(row.get("mime_type", "")).strip().lower()
    if mime_type.startswith("video/"):
        return True

    extension = str(row.get("extension", "")).strip().lower()
    if extension and not extension.startswith("."):
        extension = f".{extension}"
    return extension in VIDEO_EXTENSIONS


def main(index_dir: Path | None = None) -> None:
    input_path, output_path = resolve_io_paths(index_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Не найден файл: {input_path}")

    df = pd.read_csv(input_path)
    if "file_path" not in df.columns:
        raise ValueError(f"Файл {input_path} должен содержать колонку file_path.")

    video_df = df[df.apply(is_video_row, axis=1)].copy()

    if "asset_id" in video_df.columns:
        video_df = video_df.drop_duplicates(subset=["asset_id"])
    else:
        video_df = video_df.drop_duplicates(subset=["file_path"])

    placeholder_columns = {
        "duration_sec": pd.NA,
        "fps": pd.NA,
        "bitrate_kbps": pd.NA,
        "audio_present": pd.NA,
        "video_codec": pd.NA,
        "audio_codec": pd.NA,
        "video_index_status": "indexed",
    }
    for column, default in placeholder_columns.items():
        if column not in video_df.columns:
            video_df[column] = default

    preferred_columns = [
        "asset_id",
        "file_path",
        "primary_file_path",
        "file_name",
        "extension",
        "file_size",
        "created_at_fs",
        "sha256",
        "width",
        "height",
        "exif_datetime",
        "json_datetime",
        "json_path",
        "album_path",
        "mime_type",
        "is_video",
        "sidecar_paths",
        "sidecar_count",
        "has_sidecar",
        "content_type_file",
        "duration_sec",
        "fps",
        "bitrate_kbps",
        "audio_present",
        "video_codec",
        "audio_codec",
        "video_index_status",
    ]
    ordered_columns = [column for column in preferred_columns if column in video_df.columns]
    trailing_columns = [column for column in video_df.columns if column not in ordered_columns]
    video_df = video_df[ordered_columns + trailing_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("video_assets =", len(video_df))
    print("video_index_saved_to =", output_path)


if __name__ == "__main__":
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(arg)
