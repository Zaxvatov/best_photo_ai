from pathlib import Path

import pandas as pd

try:
    import config_paths
except ImportError as e:
    raise ImportError("Не удалось импортировать config_paths.py") from e


REQUIRED_PATHS = [
    "UNIQUE_MEDIA",
    "ANALYSIS_IMAGES",
    "PHOTO_INDEX",
]

available_paths = [name for name in dir(config_paths) if name.isupper()]
missing_paths = [name for name in REQUIRED_PATHS if not hasattr(config_paths, name)]
if missing_paths:
    raise ImportError(
        "config_paths.py не содержит: "
        + ", ".join(missing_paths)
        + ". Доступные переменные: "
        + str(available_paths)
    )


INPUT = Path(config_paths.UNIQUE_MEDIA)
OUT = Path(config_paths.ANALYSIS_IMAGES)
OUT_PHOTO_INDEX = Path(config_paths.PHOTO_INDEX)


REQUIRED_COLUMNS = [
    "asset_id",
    "primary_file_path",
    "is_image",
    "phash",
]


def validate_input_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns in unique_media: "
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

    # Keep the current file_path-based downstream contract, but ensure the
    # asset identity travels with every analysis row.
    imgs["file_path"] = imgs["primary_file_path"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    imgs.to_csv(OUT, index=False, encoding="utf-8-sig")
    imgs.to_csv(OUT_PHOTO_INDEX, index=False, encoding="utf-8-sig")

    print("rows =", len(df))
    print("images_for_analysis =", len(imgs))
    print("saved_to =", OUT)
    print("photo_index_saved_to =", OUT_PHOTO_INDEX)


if __name__ == "__main__":
    main()
