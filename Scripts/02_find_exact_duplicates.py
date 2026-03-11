import pandas as pd
from pathlib import Path

INPUT = r"D:\photo_ai\data\index\media_index.csv"
OUT_UNIQUE = r"D:\photo_ai\data\index\unique_media.csv"
OUT_DUPLICATES = r"D:\photo_ai\data\index\exact_duplicates.csv"


def main():
    df = pd.read_csv(INPUT)

    dup = df[df["sha256"].duplicated(keep=False)].copy()
    dup = dup.sort_values(["sha256", "file_path"])
    dup.to_csv(OUT_DUPLICATES, index=False, encoding="utf-8-sig")

    unique = df.drop_duplicates(subset=["sha256"], keep="first").copy()
    unique.to_csv(OUT_UNIQUE, index=False, encoding="utf-8-sig")

    print("rows =", len(df))
    print("duplicate_rows =", len(dup))
    print("duplicate_groups =", dup["sha256"].nunique() if len(dup) else 0)
    print("unique_files =", len(unique))
    print("saved_unique_to =", OUT_UNIQUE)
    print("saved_duplicates_to =", OUT_DUPLICATES)


if __name__ == "__main__":
    main()