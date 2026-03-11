import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

INPUT = r"D:\photo_ai\data\index\similar_groups.csv"
OUT = r"D:\photo_ai\data\index\sharpness.csv"


def compute_sharpness(path: Path):
    try:
        img = np.array(Image.open(path).convert("L"))
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        return None


def main():
    df = pd.read_csv(INPUT)

    rows = []
    for p in df["file_path"]:
        rows.append((p, compute_sharpness(Path(p))))

    out = pd.DataFrame(rows, columns=["file_path", "sharpness"])
    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    print("processed =", len(out))
    print("nan =", int(out["sharpness"].isna().sum()))
    print("saved_to =", OUT)


if __name__ == "__main__":
    main()