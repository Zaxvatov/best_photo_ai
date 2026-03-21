from pathlib import Path
import sys
import os
from concurrent.futures import ThreadPoolExecutor

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
    "PHOTO_INDEX",
    "SHARPNESS",
)
WORKERS = max(2, min(16, (os.cpu_count() or 8)))


def validate_config() -> None:
    missing = [name for name in REQUIRED_CONFIG_ATTRS if not hasattr(cfg, name)]
    if missing:
        available = sorted(name for name in dir(cfg) if name.isupper())
        raise ImportError(
            "config_paths.py не содержит: "
            + ", ".join(missing)
            + ". Доступные переменные: "
            + str(available)
        )


def resolve_io_paths(index_dir: Path | None) -> tuple[Path, Path]:
    validate_config()

    if index_dir is None:
        return Path(cfg.PHOTO_INDEX), Path(cfg.SHARPNESS)

    return index_dir / Path(cfg.PHOTO_INDEX).name, index_dir / Path(cfg.SHARPNESS).name


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


def compute_row(row: dict[str, object], path_column: str) -> dict[str, object]:
    image_path_str = str(row[path_column])
    image_path = Path(image_path_str)
    result = {
        path_column: image_path_str,
        "sharpness": compute_sharpness(image_path),
    }
    if "asset_id" in row:
        result["asset_id"] = row["asset_id"]
    return result


def main(index_dir: Path | None = None) -> None:
    photo_index_path, output_path = resolve_io_paths(index_dir)

    if not photo_index_path.exists():
        raise FileNotFoundError(f"Не найден файл: {photo_index_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(photo_index_path)

    path_column = None
    for candidate in ("file_path", "path", "full_path"):
        if candidate in df.columns:
            path_column = candidate
            break

    if path_column is None:
        raise KeyError(
            "Во входном файле не найдена колонка с путём к файлу. "
            "Ожидалась одна из: file_path, path, full_path"
        )

    rows = df[[path_column] + (["asset_id"] if "asset_id" in df.columns else [])].to_dict("records")
    print(f"workers = {WORKERS}")
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        results = list(
            tqdm(
                executor.map(lambda row: compute_row(row, path_column), rows),
                total=len(rows),
                desc="Sharpness",
                unit="image",
            )
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"rows = {len(df)}")
    print(f"nan = {int(result_df['sharpness'].isna().sum())}")
    print(f"saved_to = {output_path}")


if __name__ == "__main__":
    arg_index_dir = Path(sys.argv[1]) if len(sys.argv) == 2 else None
    main(arg_index_dir)
