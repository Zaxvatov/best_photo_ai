from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config_paths as cfg


REQUIRED_CONFIG_ATTRS = (
    "INDEX_DIR",
)


def _cfg_path(name: str) -> Path | None:
    value = getattr(cfg, name, None)
    return Path(value) if value else None


def validate_config() -> None:
    missing = [name for name in REQUIRED_CONFIG_ATTRS if not hasattr(cfg, name)]
    if missing:
        raise AttributeError(
            "В config_paths.py отсутствуют обязательные записи: " + ", ".join(missing)
        )


def resolve_io_paths(index_dir: Path | None) -> tuple[Path, Path]:
    validate_config()

    base_index_dir = index_dir if index_dir is not None else Path(cfg.INDEX_DIR)

    analysis_csv = (
        _cfg_path("ANALYSIS_IMAGES_CSV")
        or _cfg_path("IN_ANALYSIS_IMAGES")
        or base_index_dir / getattr(cfg, "ANALYSIS_IMAGES_NAME", "analysis_images.csv")
    )

    output_csv = (
        _cfg_path("SHARPNESS_SCORES_CSV")
        or _cfg_path("OUT_SHARPNESS_SCORES")
        or _cfg_path("SHARPNESS_CSV")
        or _cfg_path("OUT_SHARPNESS")
        or base_index_dir / getattr(cfg, "SHARPNESS_SCORES_NAME", "sharpness_scores.csv")
    )

    return analysis_csv, output_csv


def compute_sharpness(image_path: Path) -> float:
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        if data.size == 0:
            return float("nan")

        image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return float("nan")

        return float(cv2.Laplacian(image, cv2.CV_64F).var())
    except Exception:
        return float("nan")


def main(index_dir: Path | None = None) -> None:
    analysis_csv, output_csv = resolve_io_paths(index_dir)

    if not analysis_csv.exists():
        raise FileNotFoundError(f"Не найден файл: {analysis_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(analysis_csv)

    path_column = None
    for candidate in ("file_path", "path", "full_path"):
        if candidate in df.columns:
            path_column = candidate
            break

    if path_column is None:
        raise KeyError(
            "В analysis_images.csv не найдена колонка с путём к файлу. "
            "Ожидалась одна из: file_path, path, full_path"
        )

    results = []

    for image_path_str in tqdm(df[path_column].tolist(), total=len(df), desc="Sharpness", unit="image"):
        image_path = Path(image_path_str)
        sharpness = compute_sharpness(image_path)
        results.append({
            path_column: image_path_str,
            "sharpness_score": sharpness,
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"rows = {len(df)}")
    print(f"nan = {int(result_df['sharpness_score'].isna().sum())}")
    print(f"saved_to = {output_csv}")


if __name__ == "__main__":
    arg_index_dir = Path(sys.argv[1]) if len(sys.argv) == 2 else None
    main(arg_index_dir)
