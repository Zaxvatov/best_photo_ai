from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config_paths as cfg


REQUIRED_CFG_VARS = [
    "VIDEO_GROUPS",
    "VIDEO_BEST",
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


IN_VIDEO_GROUPS = Path(cfg.VIDEO_GROUPS)
OUT_VIDEO_BEST = Path(cfg.VIDEO_BEST)
IN_LIVE_PHOTO = Path(cfg.LIVE_PHOTO_CANDIDATES)


def norm_by_group(series: pd.Series, groups: pd.Series) -> pd.Series:
    maxv = series.groupby(groups).transform(lambda x: x.max() if x.max() > 0 else 1)
    return (series / maxv).clip(lower=0, upper=1)


def prepare_numeric(df: pd.DataFrame, column: str, fallback_column: str | None = None) -> pd.Series:
    if column in df.columns:
        base = pd.to_numeric(df[column], errors="coerce")
        if fallback_column and fallback_column in df.columns:
            fallback = pd.to_numeric(df[fallback_column], errors="coerce")
            base = base.fillna(fallback)
        return base.fillna(0.0)
    if fallback_column and fallback_column in df.columns:
        return pd.to_numeric(df[fallback_column], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def prepare_bool(df: pd.DataFrame, column: str, fallback_column: str | None = None) -> pd.Series:
    source = None
    if column in df.columns:
        source = df[column]
    elif fallback_column and fallback_column in df.columns:
        source = df[fallback_column]
    else:
        return pd.Series(0.0, index=df.index)

    normalized = (
        source.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
        .fillna(0.0)
    )
    return normalized


def main() -> None:
    if not IN_VIDEO_GROUPS.exists():
        raise FileNotFoundError(f"Не найден файл: {IN_VIDEO_GROUPS}")

    df = pd.read_csv(IN_VIDEO_GROUPS)
    if df.empty:
        raise ValueError(f"Файл {IN_VIDEO_GROUPS} пуст.")
    if "asset_id" not in df.columns or "video_group_id" not in df.columns:
        raise KeyError("video_groups.csv должен содержать asset_id и video_group_id.")

    df = df.copy()
    df["duration_sec_final"] = prepare_numeric(df, "duration_sec", "duration_sec_metric")
    df["fps_final"] = prepare_numeric(df, "fps", "fps_metric")
    df["width_final"] = prepare_numeric(df, "width", "width_metric")
    df["height_final"] = prepare_numeric(df, "height", "height_metric")
    df["bitrate_kbps_final"] = prepare_numeric(df, "bitrate_kbps", "bitrate_kbps_metric")
    df["audio_present_final"] = prepare_bool(df, "audio_present", "audio_present_metric")
    df["sidecar_count_final"] = prepare_numeric(df, "sidecar_count")
    df["pixels_final"] = (df["width_final"] * df["height_final"]).fillna(0.0)

    if IN_LIVE_PHOTO.exists():
        live_df = pd.read_csv(IN_LIVE_PHOTO)
        if not live_df.empty and "video_asset_id" in live_df.columns:
            live_df = (
                live_df.sort_values(["video_asset_id", "time_gap_sec"])
                .drop_duplicates(subset=["video_asset_id"], keep="first")
                .rename(columns={"video_asset_id": "asset_id"})
            )
            df = df.merge(
                live_df[["asset_id", "live_photo_status", "time_gap_sec"]],
                on="asset_id",
                how="left",
            )

    if "live_photo_status" not in df.columns:
        df["live_photo_status"] = pd.NA
    if "time_gap_sec" not in df.columns:
        df["time_gap_sec"] = pd.NA

    df["has_live_photo_pair"] = df["live_photo_status"].notna()
    df["live_photo_penalty"] = (
        df["live_photo_status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"strong": 0.20, "probable": 0.10})
        .fillna(0.0)
    )

    df["duration_norm"] = norm_by_group(df["duration_sec_final"], df["video_group_id"])
    df["fps_norm"] = norm_by_group(df["fps_final"], df["video_group_id"])
    df["pixels_norm"] = norm_by_group(df["pixels_final"], df["video_group_id"])
    df["bitrate_norm"] = norm_by_group(df["bitrate_kbps_final"], df["video_group_id"])
    df["sidecar_norm"] = norm_by_group(df["sidecar_count_final"], df["video_group_id"])

    df["video_score_raw"] = (
        0.35 * df["pixels_norm"]
        + 0.25 * df["bitrate_norm"]
        + 0.12 * df["fps_norm"]
        + 0.08 * df["duration_norm"]
        + 0.10 * df["audio_present_final"]
        + 0.05 * df["sidecar_norm"]
        + 0.05 * (df["video_metrics_status"].astype(str).eq("ok").astype(float))
    )
    df["video_score"] = (df["video_score_raw"] - df["live_photo_penalty"]).clip(lower=0.0).round(6)

    best_by_group = df.groupby("video_group_id")["video_score"].transform("max")
    df["is_best_video"] = df["video_score"].eq(best_by_group)

    tie_break_order = [
        "video_group_id",
        "is_best_video",
        "video_score",
        "pixels_final",
        "bitrate_kbps_final",
        "duration_sec_final",
        "fps_final",
        "asset_id",
    ]
    ascending = [True, False, False, False, False, False, False, True]
    best_assets = (
        df.sort_values(tie_break_order, ascending=ascending)
        .drop_duplicates(subset=["video_group_id"], keep="first")
        [["video_group_id", "asset_id"]]
        .rename(columns={"asset_id": "best_asset_id"})
    )
    df = df.drop(columns=["is_best_video"]).merge(best_assets, on="video_group_id", how="left")
    df["is_best_video"] = df["asset_id"].eq(df["best_asset_id"])

    preferred_columns = [
        "video_group_id",
        "video_group_size",
        "asset_id",
        "best_asset_id",
        "is_best_video",
        "video_score",
        "video_score_raw",
        "live_photo_penalty",
        "has_live_photo_pair",
        "live_photo_status",
        "time_gap_sec",
        "duration_norm",
        "fps_norm",
        "pixels_norm",
        "bitrate_norm",
        "sidecar_norm",
        "primary_file_path",
        "file_name",
        "album_path",
        "duration_sec_final",
        "fps_final",
        "width_final",
        "height_final",
        "bitrate_kbps_final",
        "audio_present_final",
        "video_codec",
        "audio_codec",
        "video_metrics_status",
        "video_group_status",
    ]
    ordered_columns = [column for column in preferred_columns if column in df.columns]
    trailing_columns = [column for column in df.columns if column not in ordered_columns]
    out_df = df[ordered_columns + trailing_columns].copy()

    OUT_VIDEO_BEST.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_VIDEO_BEST, index=False, encoding="utf-8-sig")

    print(f"groups_processed = {int(df['video_group_id'].nunique())}")
    print(f"best_rows = {int(df['is_best_video'].sum())}")
    print(f"saved_to = {OUT_VIDEO_BEST}")


if __name__ == "__main__":
    main()
