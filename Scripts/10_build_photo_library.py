from pathlib import Path
import shutil
import pandas as pd

INDEX_DIR = Path(r"D:\photo_ai\data\index")
OUT_DIR = Path(r"D:\photo_ai\data\output")

IN_BEST = INDEX_DIR / "best_combined.csv"
IN_REVIEW = INDEX_DIR / "review_groups.csv"
IN_GROUPS = INDEX_DIR / "similar_groups.csv"


def safe_name(p: str) -> str:
    return Path(p).name


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    best = pd.read_csv(IN_BEST)
    review = pd.read_csv(IN_REVIEW)
    groups = pd.read_csv(IN_GROUPS)

    best_dir = OUT_DIR / "best"
    review_dir = OUT_DIR / "review"
    best_dir.mkdir(exist_ok=True)
    review_dir.mkdir(exist_ok=True)

    group_to_best = dict(zip(best["group_id"], best["file_path"]))

    copied_best = 0
    copied_review = 0

    for _, row in best.iterrows():
        src = Path(row["file_path"])
        if src.exists():
            dst = best_dir / safe_name(str(src))
            shutil.copy2(src, dst)
            copied_best += 1

    review_group_ids = set(review["group_id"].dropna().tolist()) if "group_id" in review.columns else set()

    if review_group_ids:
        review_rows = groups[groups["group_id"].isin(review_group_ids)].copy()
        for _, row in review_rows.iterrows():
            src = Path(row["file_path"])
            if not src.exists():
                continue
            group_id = row["group_id"]
            group_dir = review_dir / f"group_{group_id}"
            group_dir.mkdir(exist_ok=True)
            dst = group_dir / safe_name(str(src))
            shutil.copy2(src, dst)
            copied_review += 1

    print(f"best_source = {IN_BEST}")
    print(f"review_source = {IN_REVIEW}")
    print(f"groups_source = {IN_GROUPS}")
    print(f"copied_best = {copied_best}")
    print(f"copied_review = {copied_review}")
    print(f"best_dir = {best_dir}")
    print(f"review_dir = {review_dir}")


if __name__ == "__main__":
    main()
