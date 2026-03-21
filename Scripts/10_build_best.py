import pandas as pd
from pathlib import Path

import config_paths as cfg


REQUIRED_CFG_VARS = [
    "SIMILAR_GROUPS",
    "SHARPNESS",
    "COMPOSITION",
    "SUBJECT",
    "AESTHETIC",
    "MEDIA_INDEX",
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
IN_MEDIA = Path(cfg.MEDIA_INDEX)
OUT_PHOTO_FEATURES = Path(cfg.PHOTO_FEATURES)
OUT_PHOTO_SEMANTIC = Path(cfg.PHOTO_SEMANTIC_SCORES)
OUT_BEST = Path(cfg.BEST_COMBINED)
OUT_REVIEW = Path(cfg.REVIEW_GROUPS)

IN_FACE_METRICS = Path(cfg.FACE_METRICS) if hasattr(cfg, "FACE_METRICS") else Path(cfg.INDEX_DIR) / "face_metrics.csv"


def norm_by_group(series, groups):
    maxv = series.groupby(groups).transform(lambda x: x.max() if x.max() > 0 else 1)
    return (series / maxv).clip(lower=0, upper=1)


def minmax_by_group(series, groups):
    mn = series.groupby(groups).transform("min")
    mx = series.groupby(groups).transform("max")
    denom = (mx - mn).replace(0, 1)
    return ((series - mn) / denom).clip(lower=0, upper=1)


def load_subject_data():
    if IN_FACE_METRICS.exists():
        df = pd.read_csv(IN_FACE_METRICS)
        if {"file_path", "faces", "face_symmetry"}.issubset(df.columns):
            return df, "face_metrics", str(IN_FACE_METRICS)

    df = pd.read_csv(IN_SUBJECT)
    if {"file_path", "subject_score"}.issubset(df.columns):
        return df, "subject_scores", str(IN_SUBJECT)

    raise ValueError(
        f"Файл {IN_SUBJECT} должен содержать колонки file_path и subject_score. "
        f"Найдено: {list(df.columns)}"
    )


def join_tags(values) -> str:
    tags = sorted({str(v).strip() for v in values if str(v).strip() not in {"", "0", "nan", "None"}})
    return ", ".join(tags)


def merge_metric(df: pd.DataFrame, metric_df: pd.DataFrame, prefer_asset: bool = True) -> pd.DataFrame:
    metric = metric_df.copy()
    if prefer_asset and "asset_id" in df.columns and "asset_id" in metric.columns:
        drop_cols = [column for column in metric.columns if column in df.columns and column not in {"asset_id"}]
        if drop_cols:
            metric = metric.drop(columns=drop_cols)
        metric = metric.drop_duplicates(subset=["asset_id"])
        return df.merge(metric, on="asset_id", how="left")
    if "file_path" not in metric.columns:
        raise KeyError("Metric dataframe must contain file_path when asset_id merge is unavailable.")
    drop_cols = [column for column in metric.columns if column in df.columns and column not in {"file_path"}]
    if drop_cols:
        metric = metric.drop(columns=drop_cols)
    metric = metric.drop_duplicates(subset=["file_path"])
    return df.merge(metric, on="file_path", how="left")


def save_photo_artifacts(df: pd.DataFrame) -> None:
    feature_cols = [
        column
        for column in [
            "asset_id",
            "file_path",
            "primary_file_path",
            "width",
            "height",
            "file_size",
            "pixels",
            "content_type_file",
            "sharpness",
            "subject_placement",
            "face_coverage",
            "edge_penalty",
            "tilt_score",
            "composition_score",
            "subject_score",
        ]
        if column in df.columns
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
        if column in df.columns
    ]

    feature_df = df[feature_cols].drop_duplicates(subset=["asset_id"] if "asset_id" in df.columns else ["file_path"]).copy()
    semantic_df = df[semantic_cols].drop_duplicates(subset=["asset_id"] if "asset_id" in df.columns else ["file_path"]).copy()

    OUT_PHOTO_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(OUT_PHOTO_FEATURES, index=False, encoding="utf-8-sig")
    semantic_df.to_csv(OUT_PHOTO_SEMANTIC, index=False, encoding="utf-8-sig")


def main():
    g = pd.read_csv(IN_GROUPS)
    s = pd.read_csv(IN_SHARPNESS)
    c = pd.read_csv(IN_COMPOSITION)
    subj, subject_mode, subject_source = load_subject_data()
    a = pd.read_csv(IN_AESTHETIC)
    desired_media_cols = [
        "file_path",
        "asset_id",
        "primary_file_path",
        "sidecar_paths",
        "sidecar_count",
        "has_sidecar",
        "width",
        "height",
        "file_size",
        "json_path",
    ]
    media_df = pd.read_csv(IN_MEDIA)
    if "content_type_file" in media_df.columns:
        desired_media_cols.append("content_type_file")
    media_cols = [
        column
        for column in desired_media_cols
        if column in media_df.columns and (column == "file_path" or column not in g.columns)
    ]
    m = media_df[media_cols]

    df = g.copy()
    df = merge_metric(df, s)
    df = merge_metric(df, c)
    df = merge_metric(df, subj)
    df = merge_metric(df, a)
    df = merge_metric(df, m, prefer_asset=False)

    df = df.fillna(0)
    df["pixels"] = df["width"] * df["height"]
    if "content_type_file" not in df.columns:
        df["content_type_file"] = ""

    if "scene_group_id" in df.columns:
        scene_sizes = df.groupby("scene_group_id")["file_path"].transform("size")
        scene_group_counts = df.groupby("scene_group_id")["group_id"].transform("nunique")
        df["scene_group_size"] = scene_sizes
        df["scene_group_group_count"] = scene_group_counts
        df["scene_merge_candidate"] = scene_group_counts > 1
        df["content_type_scene"] = df.groupby("scene_group_id")["content_type_file"].transform(join_tags)
    else:
        df["scene_group_size"] = 1
        df["scene_group_group_count"] = 1
        df["scene_merge_candidate"] = False
        df["content_type_scene"] = df["content_type_file"]

    df["content_type_group"] = df.groupby("group_id")["content_type_file"].transform(join_tags)

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

    save_photo_artifacts(df)

    best = df.sort_values("final_score", ascending=False).groupby("group_id").first().reset_index()
    best.to_csv(OUT_BEST, index=False, encoding="utf-8-sig")

    best_key_cols = ["group_id", "file_path", "final_score"]
    rename_map = {"file_path": "best_file", "final_score": "best_score"}
    if "asset_id" in best.columns:
        best_key_cols.append("asset_id")
        rename_map["asset_id"] = "best_asset_id"

    review = df.merge(
        best[best_key_cols].rename(columns=rename_map),
        on="group_id",
        how="left"
    )
    review["is_best"] = review["file_path"] == review["best_file"]
    review.to_csv(OUT_REVIEW, index=False, encoding="utf-8-sig")

    print("sharpness_source =", IN_SHARPNESS)
    print("subject_mode =", subject_mode)
    print("subject_source =", subject_source)
    print("photo_features_saved_to =", OUT_PHOTO_FEATURES)
    print("photo_semantic_saved_to =", OUT_PHOTO_SEMANTIC)
    print("groups_processed =", len(best))
    print("best_saved_to =", OUT_BEST)
    print("review_saved_to =", OUT_REVIEW)


if __name__ == "__main__":
    main()
