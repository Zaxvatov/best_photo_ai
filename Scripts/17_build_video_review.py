from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config_paths as cfg


REQUIRED_CFG_VARS = [
    "VIDEO_BEST",
    "VIDEO_REVIEW_GROUPS",
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


IN_VIDEO_BEST = Path(cfg.VIDEO_BEST)
OUT_VIDEO_REVIEW = Path(cfg.VIDEO_REVIEW_GROUPS)
IN_LIVE_PHOTO = Path(cfg.LIVE_PHOTO_CANDIDATES) if hasattr(cfg, "LIVE_PHOTO_CANDIDATES") else None


def main() -> None:
    if not IN_VIDEO_BEST.exists():
        raise FileNotFoundError(f"Не найден файл: {IN_VIDEO_BEST}")

    df = pd.read_csv(IN_VIDEO_BEST)
    if df.empty:
        raise ValueError(f"Файл {IN_VIDEO_BEST} пуст.")

    if "video_group_id" not in df.columns:
        raise KeyError("video_best.csv должен содержать video_group_id.")

    review_df = df.copy()
    review_df["group_id"] = review_df["video_group_id"]
    review_df["scene_group_id"] = review_df["video_group_id"]
    review_df["is_best"] = review_df.get("is_best_video", False)
    review_df["final_score"] = review_df.get("video_score", 0.0)
    review_df["media_type"] = "video"
    review_df["content_type_file"] = review_df.get("content_type_file", "video").fillna("video")
    review_df["content_type_group"] = "video"
    review_df["content_type_scene"] = "video"
    review_df["scene_group_size"] = review_df.get("video_group_size", 1)
    review_df["best_file"] = review_df.groupby("group_id")["file_name"].transform("first")
    review_df["best_score"] = review_df.groupby("group_id")["final_score"].transform("max")

    if IN_LIVE_PHOTO is not None and IN_LIVE_PHOTO.exists():
        live_df = pd.read_csv(IN_LIVE_PHOTO)
        if not live_df.empty and "video_asset_id" in live_df.columns:
            live_df = (
                live_df.sort_values(["video_asset_id", "time_gap_sec"])
                .drop_duplicates(subset=["video_asset_id"], keep="first")
                .rename(
                    columns={
                        "video_asset_id": "asset_id",
                        "photo_asset_id": "live_photo_photo_asset_id",
                        "photo_file_name": "live_photo_photo_file_name",
                        "photo_primary_file_path": "live_photo_photo_primary_file_path",
                    }
                )
            )
            review_df = review_df.merge(
                live_df[
                    [
                        "asset_id",
                        "live_photo_photo_asset_id",
                        "live_photo_photo_file_name",
                        "live_photo_photo_primary_file_path",
                        "time_gap_sec",
                        "live_photo_status",
                    ]
                ],
                on="asset_id",
                how="left",
            )
            review_df["has_live_photo_pair"] = review_df["live_photo_status"].notna()
        else:
            review_df["has_live_photo_pair"] = False
    else:
        review_df["has_live_photo_pair"] = False

    preferred_columns = [
        "group_id",
        "scene_group_id",
        "scene_group_size",
        "asset_id",
        "best_asset_id",
        "file_path",
        "primary_file_path",
        "file_name",
        "album_path",
        "media_type",
        "is_best",
        "final_score",
        "best_file",
        "best_score",
        "video_score",
        "video_score_raw",
        "live_photo_penalty",
        "duration_sec_final",
        "fps_final",
        "width_final",
        "height_final",
        "bitrate_kbps_final",
        "audio_present_final",
        "video_codec",
        "audio_codec",
        "has_live_photo_pair",
        "live_photo_status",
        "live_photo_photo_file_name",
        "live_photo_photo_primary_file_path",
        "time_gap_sec",
        "video_metrics_status",
        "video_metrics_backend",
        "video_group_status",
        "content_type_file",
        "content_type_group",
        "content_type_scene",
        "sidecar_paths",
        "sidecar_count",
        "has_sidecar",
        "json_path",
    ]
    ordered_columns = [column for column in preferred_columns if column in review_df.columns]
    trailing_columns = [column for column in review_df.columns if column not in ordered_columns]
    review_df = review_df[ordered_columns + trailing_columns]

    OUT_VIDEO_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(OUT_VIDEO_REVIEW, index=False, encoding="utf-8-sig")

    print(f"groups_processed = {int(review_df['group_id'].nunique())}")
    print(f"rows = {len(review_df)}")
    print(f"saved_to = {OUT_VIDEO_REVIEW}")


if __name__ == "__main__":
    main()
