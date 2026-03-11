import pandas as pd
from pathlib import Path

IN_GROUPS = r"D:\photo_ai\data\index\similar_groups.csv"
IN_SHARPNESS = r"D:\photo_ai\data\index\sharpness.csv"
IN_COMPOSITION = r"D:\photo_ai\data\index\composition_scores.csv"
IN_FACE_METRICS = r"D:\photo_ai\data\index\face_metrics.csv"
IN_AESTHETIC = r"D:\photo_ai\data\index\aesthetic_scores.csv"
IN_MEDIA = r"D:\photo_ai\data\index\media_index.csv"

OUT_BEST = r"D:\photo_ai\data\index\best_combined_v7.csv"
OUT_REVIEW = r"D:\photo_ai\data\index\review_groups.csv"


def norm_by_group(series, groups):
    maxv = series.groupby(groups).transform(lambda x: x.max() if x.max() > 0 else 1)
    return (series / maxv).clip(lower=0, upper=1)


def minmax_by_group(series, groups):
    mn = series.groupby(groups).transform("min")
    mx = series.groupby(groups).transform("max")
    denom = (mx - mn).replace(0, 1)
    return ((series - mn) / denom).clip(lower=0, upper=1)


def main():
    g = pd.read_csv(IN_GROUPS)
    s = pd.read_csv(IN_SHARPNESS)
    c = pd.read_csv(IN_COMPOSITION)
    fm = pd.read_csv(IN_FACE_METRICS)
    a = pd.read_csv(IN_AESTHETIC)
    m = pd.read_csv(IN_MEDIA)[["file_path", "width", "height", "file_size", "json_path"]]

    df = (
        g.merge(s, on="file_path", how="left")
         .merge(c, on="file_path", how="left")
         .merge(fm, on="file_path", how="left")
         .merge(a, on="file_path", how="left")
         .merge(m, on="file_path", how="left")
    )

    df = df.fillna(0)
    df["pixels"] = df["width"] * df["height"]

    df["sharpness_n"] = norm_by_group(df["sharpness"], df["group_id"])
    df["faces_n"] = norm_by_group(df["faces"], df["group_id"])
    df["pixels_n"] = norm_by_group(df["pixels"], df["group_id"])
    df["file_size_n"] = norm_by_group(df["file_size"], df["group_id"])
    df["eyes_score"] = norm_by_group(df["eyes_open_score"], df["group_id"])
    df["smile_n"] = norm_by_group(df["smile_score"], df["group_id"])
    df["yaw_score"] = (1 - df["head_yaw"]).clip(lower=0, upper=1)
    df["symmetry_score"] = df["face_symmetry"].clip(lower=0, upper=1)
    df["gaze_camera_score"] = df["gaze_to_camera_score"].clip(lower=0, upper=1)
    df["aesthetic_n"] = minmax_by_group(df["aesthetic_score"], df["group_id"])

    df["score_v7"] = (
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

    best = df.sort_values("score_v7", ascending=False).groupby("group_id").first().reset_index()
    best.to_csv(OUT_BEST, index=False, encoding="utf-8-sig")

    review = df.merge(
        best[["group_id", "file_path", "score_v7"]].rename(
            columns={"file_path": "best_file", "score_v7": "best_score"}
        ),
        on="group_id",
        how="left"
    )
    review["is_best"] = review["file_path"] == review["best_file"]
    review.to_csv(OUT_REVIEW, index=False, encoding="utf-8-sig")

    print("groups_processed =", len(best))
    print("best_saved_to =", OUT_BEST)
    print("review_saved_to =", OUT_REVIEW)


if __name__ == "__main__":
    main()