from pathlib import Path
import re
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config_paths as cfg


REQUIRED_CFG_VARS = [
    "PHOTO_INDEX",
    "VIDEO_INDEX",
    "VIDEO_METRICS",
    "LIVE_PHOTO_CANDIDATES",
]

missing = [name for name in REQUIRED_CFG_VARS if not hasattr(cfg, name)]
if missing:
    available = sorted(name for name in dir(cfg) if name.isupper())
    raise ImportError(
        "config_paths.py не содержит: "
        + ", ".join(missing)
        + ". Доступные переменные: "
        + str(available)
    )


IN_PHOTO_INDEX = Path(cfg.PHOTO_INDEX)
IN_VIDEO_INDEX = Path(cfg.VIDEO_INDEX)
IN_VIDEO_METRICS = Path(cfg.VIDEO_METRICS)
OUT_LIVE_PHOTO = Path(cfg.LIVE_PHOTO_CANDIDATES)

MAX_LIVE_DURATION_SEC = 4.0
MAX_TIME_GAP_SEC = 5.0


def normalize_stem(value: object) -> str:
    stem = str(value).strip().lower()
    return re.sub(r"\.[^.]+$", "", stem)


def first_valid_ts(df: pd.DataFrame) -> pd.Series:
    json_dt = pd.to_numeric(df.get("json_datetime"), errors="coerce")
    fs_dt = pd.to_numeric(df.get("created_at_fs"), errors="coerce")
    return json_dt.fillna(fs_dt)


def empty_result() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "video_asset_id",
            "photo_asset_id",
            "album_path",
            "video_file_name",
            "photo_file_name",
            "video_primary_file_path",
            "photo_primary_file_path",
            "duration_sec",
            "time_gap_sec",
            "live_photo_status",
        ]
    )


def main() -> None:
    if not IN_PHOTO_INDEX.exists():
        raise FileNotFoundError(f"Не найден файл: {IN_PHOTO_INDEX}")
    if not IN_VIDEO_INDEX.exists():
        raise FileNotFoundError(f"Не найден файл: {IN_VIDEO_INDEX}")
    if not IN_VIDEO_METRICS.exists():
        raise FileNotFoundError(f"Не найден файл: {IN_VIDEO_METRICS}")

    photos = pd.read_csv(IN_PHOTO_INDEX).copy()
    videos = pd.read_csv(IN_VIDEO_INDEX).copy()
    video_metrics = pd.read_csv(IN_VIDEO_METRICS).copy()

    photos["stem_norm"] = photos["file_name"].map(normalize_stem)
    videos["stem_norm"] = videos["file_name"].map(normalize_stem)
    photos["capture_ts"] = first_valid_ts(photos)
    videos["capture_ts"] = first_valid_ts(videos)

    duration_map = (
        video_metrics[["asset_id", "duration_sec"]]
        .drop_duplicates(subset=["asset_id"])
        .rename(columns={"duration_sec": "duration_sec_metric"})
    )
    videos = videos.merge(duration_map, on="asset_id", how="left")
    videos["duration_sec_metric"] = pd.to_numeric(videos["duration_sec_metric"], errors="coerce")

    merged = videos.merge(
        photos[
            [
                "asset_id",
                "file_name",
                "primary_file_path",
                "album_path",
                "stem_norm",
                "capture_ts",
            ]
        ],
        on=["album_path", "stem_norm"],
        how="inner",
        suffixes=("_video", "_photo"),
    )

    if merged.empty:
        result = empty_result()
    else:
        merged["time_gap_sec"] = (merged["capture_ts_video"] - merged["capture_ts_photo"]).abs()
        result = merged[
            (merged["duration_sec_metric"] <= MAX_LIVE_DURATION_SEC)
            & (merged["time_gap_sec"] <= MAX_TIME_GAP_SEC)
        ].copy()

        if result.empty:
            result = empty_result()
        else:
            result["live_photo_status"] = result["time_gap_sec"].apply(
                lambda gap: "strong" if gap <= 1.0 else "probable"
            )
            result = result.rename(
                columns={
                    "asset_id_video": "video_asset_id",
                    "asset_id_photo": "photo_asset_id",
                    "file_name_video": "video_file_name",
                    "file_name_photo": "photo_file_name",
                    "primary_file_path_video": "video_primary_file_path",
                    "primary_file_path_photo": "photo_primary_file_path",
                    "duration_sec_metric": "duration_sec",
                }
            )
            result = result[
                [
                    "video_asset_id",
                    "photo_asset_id",
                    "album_path",
                    "video_file_name",
                    "photo_file_name",
                    "video_primary_file_path",
                    "photo_primary_file_path",
                    "duration_sec",
                    "time_gap_sec",
                    "live_photo_status",
                ]
            ].sort_values(["live_photo_status", "album_path", "video_file_name", "time_gap_sec"])

    OUT_LIVE_PHOTO.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_LIVE_PHOTO, index=False, encoding="utf-8-sig")

    print(f"photo_assets = {len(photos)}")
    print(f"video_assets = {len(videos)}")
    print(f"live_photo_candidates = {len(result)}")
    if not result.empty:
        strong = int((result["live_photo_status"] == "strong").sum())
        probable = int((result["live_photo_status"] == "probable").sum())
        print(f"strong = {strong}")
        print(f"probable = {probable}")
    print(f"saved_to = {OUT_LIVE_PHOTO}")


if __name__ == "__main__":
    main()
