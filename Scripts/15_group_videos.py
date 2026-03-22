from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config_paths as cfg


REQUIRED_CONFIG_ATTRS = (
    "VIDEO_INDEX",
    "VIDEO_METRICS",
    "VIDEO_GROUPS",
)

LOOKAHEAD = 8
TIME_WINDOW_STRICT_SEC = 120
TIME_WINDOW_RELAXED_SEC = 300
TIME_WINDOW_FS_ONLY_SEC = 45
NUMERIC_GAP_STRICT = 20
NUMERIC_GAP_RELAXED = 6


@dataclass
class PairDecision:
    matched: bool
    reason: str


class UnionFind:
    def __init__(self, items: list[int]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


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


def resolve_io_paths(index_dir: Path | None) -> tuple[Path, Path, Path]:
    validate_config()
    if index_dir is None:
        return Path(cfg.VIDEO_INDEX), Path(cfg.VIDEO_METRICS), Path(cfg.VIDEO_GROUPS)

    return (
        index_dir / Path(cfg.VIDEO_INDEX).name,
        index_dir / Path(cfg.VIDEO_METRICS).name,
        index_dir / Path(cfg.VIDEO_GROUPS).name,
    )


def parse_numeric_token(file_name: str) -> float | pd.NA:
    match = re.search(r"(\d{3,})", str(file_name))
    if not match:
        return pd.NA
    try:
        return float(match.group(1))
    except Exception:
        return pd.NA


def first_valid_number(*values: object) -> float | pd.NA:
    for value in values:
        try:
            numeric = float(value)
        except Exception:
            continue
        if math.isfinite(numeric) and numeric > 0:
            return numeric
    return pd.NA


def same_resolution(left: pd.Series, right: pd.Series) -> bool:
    lw = first_valid_number(left.get("width"))
    rw = first_valid_number(right.get("width"))
    lh = first_valid_number(left.get("height"))
    rh = first_valid_number(right.get("height"))
    return lw is not pd.NA and rw is not pd.NA and lh is not pd.NA and rh is not pd.NA and lw == rw and lh == rh


def same_codec(left: pd.Series, right: pd.Series) -> bool:
    left_codec = str(left.get("video_codec", "")).strip().lower()
    right_codec = str(right.get("video_codec", "")).strip().lower()
    return bool(left_codec and right_codec and left_codec == right_codec)


def choose_timestamp(row: pd.Series) -> tuple[float | pd.NA, str]:
    json_ts = first_valid_number(row.get("json_datetime"))
    if json_ts is not pd.NA:
        return json_ts, "json"
    fs_ts = first_valid_number(row.get("created_at_fs"))
    if fs_ts is not pd.NA:
        return fs_ts, "fs"
    return pd.NA, "missing"


def duration_ratio(left: pd.Series, right: pd.Series) -> float | pd.NA:
    d1 = first_valid_number(left.get("duration_sec"))
    d2 = first_valid_number(right.get("duration_sec"))
    if d1 is pd.NA or d2 is pd.NA:
        return pd.NA
    small = min(d1, d2)
    large = max(d1, d2)
    if small <= 0:
        return pd.NA
    return large / small


def should_link(left: pd.Series, right: pd.Series) -> PairDecision:
    if str(left.get("album_path", "")) != str(right.get("album_path", "")):
        return PairDecision(False, "album")

    left_ts = left.get("_sort_ts")
    right_ts = right.get("_sort_ts")
    if left_ts is pd.NA or right_ts is pd.NA:
        return PairDecision(False, "timestamp_missing")

    time_gap = abs(float(left_ts) - float(right_ts))
    left_num = left.get("_file_num")
    right_num = right.get("_file_num")
    numeric_gap = abs(float(left_num) - float(right_num)) if left_num is not pd.NA and right_num is not pd.NA else None

    left_src = str(left.get("_timestamp_source", ""))
    right_src = str(right.get("_timestamp_source", ""))
    fs_only = left_src == "fs" and right_src == "fs"

    if time_gap <= 30:
        return PairDecision(True, "time<=30")

    if not fs_only and numeric_gap is not None and numeric_gap <= NUMERIC_GAP_STRICT and time_gap <= TIME_WINDOW_STRICT_SEC:
        return PairDecision(True, "strict_time+numeric")

    if not fs_only and numeric_gap is not None and numeric_gap <= NUMERIC_GAP_RELAXED and time_gap <= TIME_WINDOW_RELAXED_SEC:
        if same_resolution(left, right) or same_codec(left, right):
            return PairDecision(True, "relaxed_time+numeric+tech")

    if fs_only and numeric_gap is not None and numeric_gap <= NUMERIC_GAP_RELAXED and time_gap <= TIME_WINDOW_FS_ONLY_SEC:
        if same_resolution(left, right):
            return PairDecision(True, "fs_only")

    ratio = duration_ratio(left, right)
    if ratio is not pd.NA and ratio <= 1.35 and time_gap <= 20 and same_resolution(left, right):
        return PairDecision(True, "time+duration+resolution")

    return PairDecision(False, "no_match")


def main(index_dir: Path | None = None) -> None:
    video_index_path, video_metrics_path, output_path = resolve_io_paths(index_dir)

    if not video_index_path.exists():
        raise FileNotFoundError(f"Не найден файл: {video_index_path}")
    if not video_metrics_path.exists():
        raise FileNotFoundError(f"Не найден файл: {video_metrics_path}")

    video_index = pd.read_csv(video_index_path)
    video_metrics = pd.read_csv(video_metrics_path)

    if "asset_id" not in video_index.columns or "asset_id" not in video_metrics.columns:
        raise KeyError("Ожидалась колонка asset_id во входных video-артефактах.")

    df = video_index.merge(video_metrics, on="asset_id", how="left", suffixes=("", "_metric"))
    df["primary_file_path"] = df.get("primary_file_path", df.get("file_path"))
    df["_file_num"] = df["file_name"].astype(str).map(parse_numeric_token)

    timestamps = df.apply(choose_timestamp, axis=1, result_type="expand")
    df["_sort_ts"] = timestamps[0]
    df["_timestamp_source"] = timestamps[1]

    df = df.sort_values(
        ["album_path", "_sort_ts", "_file_num", "file_name"],
        na_position="last",
    ).reset_index(drop=True)

    uf = UnionFind(list(df.index))
    links_found = 0

    for start_idx in range(len(df)):
        left = df.iloc[start_idx]
        for next_idx in range(start_idx + 1, min(start_idx + 1 + LOOKAHEAD, len(df))):
            right = df.iloc[next_idx]
            if str(left.get("album_path", "")) != str(right.get("album_path", "")):
                break

            decision = should_link(left, right)
            if decision.matched:
                uf.union(start_idx, next_idx)
                links_found += 1

    root_to_group: dict[int, int] = {}
    group_ids: list[int] = []
    next_group_id = 1
    for idx in range(len(df)):
        root = uf.find(idx)
        if root not in root_to_group:
            root_to_group[root] = next_group_id
            next_group_id += 1
        group_ids.append(root_to_group[root])

    df["video_group_id"] = group_ids
    df["video_group_size"] = df.groupby("video_group_id")["asset_id"].transform("count")
    df["video_group_status"] = "grouped"

    preferred_columns = [
        "video_group_id",
        "video_group_size",
        "asset_id",
        "primary_file_path",
        "file_name",
        "album_path",
        "json_datetime",
        "created_at_fs",
        "_timestamp_source",
        "duration_sec",
        "fps",
        "frame_count",
        "width",
        "height",
        "bitrate_kbps",
        "audio_present",
        "video_codec",
        "audio_codec",
        "video_metrics_status",
        "video_group_status",
    ]
    ordered_columns = [column for column in preferred_columns if column in df.columns]
    trailing_columns = [column for column in df.columns if column not in ordered_columns and not column.startswith("_")]
    output_df = df[ordered_columns + trailing_columns].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    groups_found = int(output_df["video_group_id"].nunique())
    files_in_multi_groups = int((output_df["video_group_size"] > 1).sum())
    print(f"video_assets = {len(output_df)}")
    print(f"links_found = {links_found}")
    print(f"groups_found = {groups_found}")
    print(f"files_in_multi_groups = {files_in_multi_groups}")
    print(f"saved_to = {output_path}")


if __name__ == "__main__":
    arg_index_dir = Path(sys.argv[1]) if len(sys.argv) == 2 else None
    main(arg_index_dir)
