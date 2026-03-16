from pathlib import Path

import pandas as pd

from config_paths import ANALYSIS_IMAGES_CSV, UNIQUE_MEDIA_CSV


INPUT = Path(UNIQUE_MEDIA_CSV)
OUT = Path(ANALYSIS_IMAGES_CSV)


REQUIRED_COLUMNS = [
    "is_image",
    "phash",
]


def validate_input_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns in unique_media.csv: "
            + ", ".join(missing)
        )


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    df = pd.read_csv(INPUT)
    validate_input_columns(df)

    imgs = df[
        (df["is_image"] == True)
        & (df["phash"].notna())
    ].copy()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    imgs.to_csv(OUT, index=False, encoding="utf-8-sig")

    print("rows =", len(df))
    print("images_for_analysis =", len(imgs))
    print("saved_to =", OUT)


if __name__ == "__main__":
    main()
