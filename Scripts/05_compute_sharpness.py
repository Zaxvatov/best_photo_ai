from pathlib import Path
import sys

import cv2
import pandas as pd
from tqdm import tqdm


def compute_sharpness(image_path: Path) -> float:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return float("nan")
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def main(index_dir: Path) -> None:
    analysis_csv = index_dir / "analysis_images.csv"
    output_csv = index_dir / "sharpness_scores.csv"

    if not analysis_csv.exists():
        raise FileNotFoundError(f"Не найден файл: {analysis_csv}")

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

    for image_path_str in tqdm(df[path_column].tolist(), desc="Sharpness", unit="image"):
        image_path = Path(image_path_str)
        sharpness = compute_sharpness(image_path)
        results.append({
            path_column: image_path_str,
            "sharpness_score": sharpness,
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)

    print(f"rows = {len(df)}")
    print(f"saved_to = {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python Scripts\\05_compute_sharpness.py <index_dir>")
        sys.exit(1)

    main(Path(sys.argv[1]))
