import pandas as pd

from config_paths import (
    MEDIA_INDEX_CSV,
    UNIQUE_MEDIA_CSV,
    EXACT_DUPLICATES_CSV,
)

INPUT = MEDIA_INDEX_CSV
OUT_UNIQUE = UNIQUE_MEDIA_CSV
OUT_DUPLICATES = EXACT_DUPLICATES_CSV


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
