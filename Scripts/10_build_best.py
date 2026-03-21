from pathlib import Path

import pandas as pd

import config_paths as cfg


REQUIRED_CFG_VARS = [
    "SIMILAR_GROUPS",
    "SHARPNESS",
    "COMPOSITION",
    "SUBJECT",
    "AESTHETIC",
    "PHOTO_INDEX",
    "PHOTO_FEATURES",
    "PHOTO_SEMANTIC_SCORES",
    "BEST_COMBINED",
    "REVIEW_GROUPS",
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


IN_GROUPS = Path(cfg.SIMILAR_GROUPS)
IN_SHARPNESS = Path(cfg.SHARPNESS)
IN_COMPOSITION = Path(cfg.COMPOSITION)
IN_SUBJECT = Path(cfg.SUBJECT)
IN_AESTHETIC = Path(cfg.AESTHETIC)
IN_PHOTO_INDEX = Path(cfg.PHOTO_INDEX)
OUT_PHOTO_FEATURES = Path(cfg.PHOTO_FEATURES)
OUT_PHOTO_SEMANTIC = Path(cfg.PHOTO_SEMANTIC_SCORES)
OUT_BEST = Path(cfg.BEST_COMBINED)
OUT_REVIEW = Path(cfg.REVIEW_GROUPS)

IN_FACE_METRICS = Path(cfg.FACE_METRICS) if hasattr(cfg, "FACE_METRICS") else Path(cfg.INDEX_DIR) / "face_metrics.csv"

FEATURE_SCORE_COLUMNS = [
    "sharpness",
    "subject_placement",
    "face_coverage",
    "edge_penalty",
    "tilt_score",
    "composition_score",
    "subject_score",
]

SEMANTIC_COLUMNS = [
    "aesthetic_score",
]


def norm_by_group(series, groups):
    maxv = series.groupby(groups).transform(lambda x: x.max() if x.max() > 0 else 1)
    return (series / maxv).clip(lower=0, upper=1)


def minmax_by_group(series, groups):
    mn = series.groupby(groups).transform("min")
    mx = series.groupby(groups).transform("max")
    denom = (mx - mn).replace(0, 1)
    return ((series - mn) / denom).clip(lower=0, upper=1)


def ensure_asset_id(metric_df: pd.DataFrame, photo_index_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if "asset_id" in metric_df.columns:
        return metric_df
    if "file_path" not in metric_df.columns:
        raise KeyError(f"{source_name} must contain asset_id or file_path.")

    asset_lookup = photo_index_df[["asset_id", "file_path"]].dropna(subset=["asset_id", "file_path"])
    asset_lookup = asset_lookup.drop_duplicates(subset=["asset_id"])
    merged = metric_df.merge(asset_lookup, on="file_path", how="left")
    if "asset_id" not in merged.columns or merged["asset_id"].isna().any():
        missing_count = int(merged["asset_id"].isna().sum()) if "asset_id" in merged.columns else len(merged)
        raise ValueError(f"{source_name} could not be mapped to asset_id for {missing_count} rows.")
    return merged


def load_subject_data(photo_index_df: pd.DataFrame):
    if IN_FACE_METRICS.exists():
        df = pd.read_csv(IN_FACE_METRICS)
        if {"file_path", "faces", "face_symmetry"}.issubset(df.columns):
            return ensure_asset_id(df, photo_index_df, str(IN_FACE_METRICS)), "face_metrics", str(IN_FACE_METRICS)

    df = pd.read_csv(IN_SUBJECT)
    if {"file_path", "subject_score"}.issubset(df.columns):
        return ensure_asset_id(df, photo_index_df, str(IN_SUBJECT)), "subject_scores", str(IN_SUBJECT)

    raise ValueError(
        f"Файл {IN_SUBJECT} должен содержать colонки file_path и subject_score. "
        f"Найдено: {list(df.columns)}"
    )


def join_tags(values) -> str:
    tags = sorted({str(v).strip() for v in values if str(v).strip() not in {"", "0", "nan", "None"}})
    return ", ".join(tags)


def dedupe_subset(df: pd.DataFrame) -> pd.DataFrame:
    if "asset_id" not in df.columns:
        raise KeyError("Expected asset_id in normalized photo pipeline dataframe.")
    return df.drop_duplicates(subset=["asset_id"])


def merge_metric(df: pd.DataFrame, metric_df: pd.DataFrame) -> pd.DataFrame:
    metric = metric_df.copy()
    if "asset_id" not in df.columns or "asset_id" not in metric.columns:
        raise KeyError("Active photo pipeline requires asset_id for metric merges.")
    drop_cols = [column for column in metric.columns if column in df.columns and column != "asset_id"]
    if drop_cols:
        metric = metric.drop(columns=drop_cols)
    metric = dedupe_subset(metric)
    return df.merge(metric, on="asset_id", how="left")


def fill_default_columns(df: pd.DataFrame, columns: list[str], default=0) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = default
    return df


def build_photo_features(photo_index_df: pd.DataFrame) -> tuple[pd.DataFrame, str, str, str]:
    feature_df = photo_index_df.copy()

    sharpness_df = ensure_asset_id(pd.read_csv(IN_SHARPNESS), photo_index_df, str(IN_SHARPNESS))
    composition_df = ensure_asset_id(pd.read_csv(IN_COMPOSITION), photo_index_df, str(IN_COMPOSITION))
    subject_df, subject_mode, subject_source = load_subject_data(photo_index_df)

    feature_df = merge_metric(feature_df, sharpness_df)
    feature_df = merge_metric(feature_df, composition_df)
    feature_df = merge_metric(feature_df, subject_df)
    feature_df = fill_default_columns(feature_df, FEATURE_SCORE_COLUMNS, 0)

    if "content_type_file" not in feature_df.columns:
        feature_df["content_type_file"] = ""

    return dedupe_subset(feature_df), str(IN_SHARPNESS), subject_mode, subject_source


def build_photo_semantic(feature_df: pd.DataFrame, groups_df: pd.DataFrame) -> pd.DataFrame:
    semantic_base_cols = [
        column
        for column in [
            "asset_id",
            "content_type_file",
            "subject_score",
        ]
        if column in feature_df.columns
    ]
    semantic_df = dedupe_subset(feature_df[semantic_base_cols].copy())

    aesthetic_df = ensure_asset_id(pd.read_csv(IN_AESTHETIC), feature_df, str(IN_AESTHETIC))
    semantic_df = merge_metric(semantic_df, aesthetic_df)
    semantic_df = fill_default_columns(semantic_df, SEMANTIC_COLUMNS, 0)

    group_cols = [
        column
        for column in [
            "asset_id",
            "group_id",
            "scene_group_id",
        ]
        if column in groups_df.columns
    ]
    grouped_assets = dedupe_subset(groups_df[group_cols].copy())
    semantic_df = merge_metric(semantic_df, grouped_assets)

    if "content_type_file" not in semantic_df.columns:
        semantic_df["content_type_file"] = ""

    semantic_df["content_type_group"] = semantic_df["content_type_file"]
    if "group_id" in semantic_df.columns:
        group_mask = semantic_df["group_id"].notna()
        semantic_df.loc[group_mask, "content_type_group"] = (
            semantic_df.loc[group_mask]
            .groupby("group_id")["content_type_file"]
            .transform(join_tags)
        )
    else:
        semantic_df["group_id"] = pd.NA

    semantic_df["content_type_scene"] = semantic_df["content_type_group"]
    if "scene_group_id" in semantic_df.columns:
        scene_mask = semantic_df["scene_group_id"].notna()
        semantic_df.loc[scene_mask, "content_type_scene"] = (
            semantic_df.loc[scene_mask]
            .groupby("scene_group_id")["content_type_file"]
            .transform(join_tags)
        )
    else:
        semantic_df["scene_group_id"] = pd.NA

    return dedupe_subset(semantic_df)


def save_photo_artifacts(feature_df: pd.DataFrame, semantic_df: pd.DataFrame) -> None:
    feature_cols = [
        column
        for column in [
            "asset_id",
            "file_path",
            "primary_file_path",
            "file_name",
            "extension",
            "file_size",
            "created_at_fs",
            "sha256",
            "phash",
            "width",
            "height",
            "exif_datetime",
            "json_datetime",
            "json_path",
            "album_path",
            "mime_type",
            "is_image",
            "is_video",
            "sidecar_paths",
            "sidecar_count",
            "has_sidecar",
            "content_type_file",
            "sharpness",
            "subject_placement",
            "face_coverage",
            "edge_penalty",
            "tilt_score",
            "composition_score",
            "subject_score",
        ]
        if column in feature_df.columns
    ]
    semantic_cols = [
        column
        for column in [
            "asset_id",
            "file_path",
            "content_type_file",
            "content_type_group",
            "content_type_scene",
            "subject_score",
            "aesthetic_score",
            "scene_group_id",
            "group_id",
        ]
        if column in semantic_df.columns
    ]

    OUT_PHOTO_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    dedupe_subset(feature_df[feature_cols].copy()).to_csv(OUT_PHOTO_FEATURES, index=False, encoding="utf-8-sig")
    dedupe_subset(semantic_df[semantic_cols].copy()).to_csv(OUT_PHOTO_SEMANTIC, index=False, encoding="utf-8-sig")


def fill_numeric_defaults(df: pd.DataFrame) -> pd.DataFrame:
    numeric_defaults = {
        "sharpness": 0,
        "subject_placement": 0,
        "face_coverage": 0,
        "edge_penalty": 0,
        "tilt_score": 0,
        "composition_score": 0,
        "subject_score": 0,
        "aesthetic_score": 0,
        "width": 0,
        "height": 0,
        "file_size": 0,
        "sidecar_count": 0,
    }
    for column, default in numeric_defaults.items():
        if column not in df.columns:
            df[column] = default
        else:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(default)

    string_defaults = [
        "content_type_file",
        "content_type_group",
        "content_type_scene",
        "primary_file_path",
        "sidecar_paths",
        "json_path",
        "album_path",
    ]
    for column in string_defaults:
        if column not in df.columns:
            df[column] = ""
        else:
            df[column] = df[column].fillna("")

    if "has_sidecar" in df.columns:
        df["has_sidecar"] = df["has_sidecar"].fillna(False)

    return df


def main():
    groups_df = pd.read_csv(IN_GROUPS)
    photo_index_df = pd.read_csv(IN_PHOTO_INDEX)

    feature_df, sharpness_source, subject_mode, subject_source = build_photo_features(photo_index_df)
    semantic_df = build_photo_semantic(feature_df, groups_df)
    save_photo_artifacts(feature_df, semantic_df)

    df = groups_df.copy()
    df = merge_metric(df, feature_df)
    df = merge_metric(df, semantic_df)
    df = fill_numeric_defaults(df)

    df["pixels"] = df["width"] * df["height"]

    if "scene_group_id" not in df.columns:
        df["scene_group_id"] = df["group_id"]

    scene_sizes = df.groupby("scene_group_id")["file_path"].transform("size")
    scene_group_counts = df.groupby("scene_group_id")["group_id"].transform("nunique")
    df["scene_group_size"] = scene_sizes
    df["scene_group_group_count"] = scene_group_counts
    df["scene_merge_candidate"] = scene_group_counts > 1

    if "content_type_scene" not in df.columns:
        df["content_type_scene"] = df["content_type_file"]
    if "content_type_group" not in df.columns:
        df["content_type_group"] = df["content_type_file"]

    df["sharpness_n"] = norm_by_group(df["sharpness"], df["group_id"])
    df["pixels_n"] = norm_by_group(df["pixels"], df["group_id"])
    df["file_size_n"] = norm_by_group(df["file_size"], df["group_id"])
    df["aesthetic_n"] = minmax_by_group(df["aesthetic_score"], df["group_id"])

    if subject_mode == "face_metrics":
        df["faces_n"] = norm_by_group(df["faces"], df["group_id"])
        df["eyes_score"] = norm_by_group(df.get("eyes_open_score", 0), df["group_id"])
        df["smile_n"] = norm_by_group(df.get("smile_score", 0), df["group_id"])
        df["yaw_score"] = (1 - df.get("head_yaw", 0)).clip(lower=0, upper=1)
        df["symmetry_score"] = df["face_symmetry"].clip(lower=0, upper=1)
        df["gaze_camera_score"] = df.get("gaze_to_camera_score", 0).clip(lower=0, upper=1)

        df["final_score"] = (
            0.22 * df["gaze_camera_score"] +
            0.14 * df["eyes_score"] +
            0.08 * df["yaw_score"] +
            0.08 * df["symmetry_score"] +
            0.10 * df["smile_n"] +
            0.14 * df["composition_score"] +
            0.14 * df["aesthetic_n"] +
            0.06 * df["sharpness_n"] +
            0.03 * df["faces_n"] +
            0.01 * df["pixels_n"]
        )
    else:
        df["subject_n"] = minmax_by_group(df["subject_score"], df["group_id"])
        df["final_score"] = (
            0.34 * df["subject_n"] +
            0.24 * df["composition_score"] +
            0.24 * df["aesthetic_n"] +
            0.14 * df["sharpness_n"] +
            0.04 * df["pixels_n"]
        )

    best = df.sort_values("final_score", ascending=False).groupby("group_id").first().reset_index()
    best.to_csv(OUT_BEST, index=False, encoding="utf-8-sig")

    best_key_cols = ["group_id", "asset_id", "file_path", "final_score"]
    rename_map = {"asset_id": "best_asset_id", "file_path": "best_file", "final_score": "best_score"}

    review = df.merge(
        best[best_key_cols].rename(columns=rename_map),
        on="group_id",
        how="left",
    )
    review["is_best"] = review["asset_id"] == review["best_asset_id"]
    review.to_csv(OUT_REVIEW, index=False, encoding="utf-8-sig")

    print("sharpness_source =", sharpness_source)
    print("subject_mode =", subject_mode)
    print("subject_source =", subject_source)
    print("photo_features_saved_to =", OUT_PHOTO_FEATURES)
    print("photo_semantic_saved_to =", OUT_PHOTO_SEMANTIC)
    print("groups_processed =", len(best))
    print("best_saved_to =", OUT_BEST)
    print("review_saved_to =", OUT_REVIEW)


if __name__ == "__main__":
    main()
