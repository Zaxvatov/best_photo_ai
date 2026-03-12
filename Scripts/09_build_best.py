import pandas as pd
from pathlib import Path

IN_GROUPS = Path(r"D:\photo_ai\data\index\similar_groups.csv")
IN_SHARPNESS = Path(r"D:\photo_ai\data\index\sharpness.csv")
IN_COMPOSITION = Path(r"D:\photo_ai\data\index\composition_scores.csv")
IN_FACE_METRICS = Path(r"D:\photo_ai\data\index\face_metrics.csv")
IN_SUBJECT = Path(r"D:\photo_ai\data\index\subject_scores.csv")
IN_AESTHETIC = Path(r"D:\photo_ai\data\index\aesthetic_scores.csv")
IN_MEDIA = Path(r"D:\photo_ai\data\index\media_index.csv")

OUT_BEST = Path(r"D:\photo_ai\data\index\best_combined.csv")
OUT_REVIEW = Path(r"D:\photo_ai\data\index\review_groups.csv")


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
            return df, "face_metrics"

    if IN_SUBJECT.exists():
        df = pd.read_csv(IN_SUBJECT)
        if {"file_path", "subject_score"}.issubset(df.columns):
            return df, "subject_scores"

    raise FileNotFoundError(
        "Не найден совместимый файл subject/face метрик. "
        f"Ожидался один из файлов: {IN_FACE_METRICS} или {IN_SUBJECT}."
    )


def main():
    g = pd.read_csv(IN_GROUPS)
    s = pd.read_csv(IN_SHARPNESS)
    c = pd.read_csv(IN_COMPOSITION)
    subj, subject_mode = load_subject_data()
    a = pd.read_csv(IN_AESTHETIC)
    m = pd.read_csv(IN_MEDIA)[["file_path", "width", "height", "file_size", "json_path"]]

    df = (
        g.merge(s, on="file_path", how="left")
         .merge(c, on="file_path", how="left")
         .merge(subj, on="file_path", how="left")
         .merge(a, on="file_path", how="left")
         .merge(m, on="file_path", how="left")
    )

    df = df.fillna(0)
    df["pixels"] = df["width"] * df["height"]

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

        df["score_v8"] = (
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

        df["score_v8"] = (
            0.34 * df["subject_n"] +
            0.24 * df["composition_score"] +
            0.24 * df["aesthetic_n"] +
            0.14 * df["sharpness_n"] +
            0.04 * df["pixels_n"]
        )

    best = df.sort_values("score_v8", ascending=False).groupby("group_id").first().reset_index()
    best.to_csv(OUT_BEST, index=False, encoding="utf-8-sig")

    review = df.merge(
        best[["group_id", "file_path", "score_v8"]].rename(
            columns={"file_path": "best_file", "score_v8": "best_score"}
        ),
        on="group_id",
        how="left"
    )
    review["is_best"] = review["file_path"] == review["best_file"]
    review.to_csv(OUT_REVIEW, index=False, encoding="utf-8-sig")

    print("subject_mode =", subject_mode)
    print("groups_processed =", len(best))
    print("best_saved_to =", OUT_BEST)
    print("review_saved_to =", OUT_REVIEW)


if __name__ == "__main__":
    main()
