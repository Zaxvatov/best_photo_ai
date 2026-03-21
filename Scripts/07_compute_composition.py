from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm

# ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import Scripts.config_paths as cfg

required_vars = ["SIMILAR_GROUPS", "COMPOSITION"]
missing = [v for v in required_vars if not hasattr(cfg, v)]

if missing:
    available = [v for v in dir(cfg) if v.isupper()]
    raise ImportError(
        f"config_paths.py не содержит: {', '.join(missing)}. Доступные переменные: {available}"
    )

SIMILAR_GROUPS = cfg.SIMILAR_GROUPS
COMPOSITION = cfg.COMPOSITION
WORKERS = max(2, min(8, max(2, (os.cpu_count() or 8) // 2)))
THREAD_LOCAL = threading.local()
MAX_ANALYSIS_DIM = 960

register_heif_opener()
cv2.setNumThreads(1)

INPUT = SIMILAR_GROUPS
OUT = COMPOSITION


def get_cascade() -> cv2.CascadeClassifier:
    cascade = getattr(THREAD_LOCAL, "cascade", None)
    if cascade is None:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        THREAD_LOCAL.cascade = cascade
    return cascade


def subject_placement_score(cx, cy):
    thirds = [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]
    d = min(np.sqrt((cx - x) ** 2 + (cy - y) ** 2) for x, y in thirds)
    return max(0, 1 - d * 1.5)


def face_coverage_score(face_area, img_area):
    r = face_area / img_area
    if 0.05 <= r <= 0.35:
        return 1.0
    if r < 0.05:
        return r / 0.05
    return max(0.0, 1 - (r - 0.35))


def edge_penalty(x, y, w, h, W, H):
    margin = 0.05
    if x < W * margin or y < H * margin:
        return 1.0
    if x + w > W * (1 - margin) or y + h > H * (1 - margin):
        return 1.0
    return 0.0


def tilt_score(img):
    edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    if lines is None:
        return 1.0

    angles = []
    for rho, theta in lines[:, 0]:
        angle = abs(theta - np.pi / 2)
        angles.append(angle)

    mean_angle = np.mean(angles)
    return max(0.0, 1 - mean_angle * 2)


def resize_for_analysis(img: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    max_dim = max(H, W)
    if max_dim <= MAX_ANALYSIS_DIM:
        return img
    scale = MAX_ANALYSIS_DIM / max_dim
    new_w = max(1, int(W * scale))
    new_h = max(1, int(H * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def process_row(row: dict[str, object]) -> dict[str, object]:
    file_path = str(row["file_path"])
    path = Path(file_path)
    result = {
        "file_path": file_path,
        "subject_placement": 0.0,
        "face_coverage": 0.0,
        "edge_penalty": 0.0,
        "tilt_score": 0.0,
        "composition_score": 0.0,
    }
    if "asset_id" in row:
        result["asset_id"] = row["asset_id"]

    try:
        img = np.array(Image.open(path).convert("L"))
        img = resize_for_analysis(img)
        H, W = img.shape

        faces = get_cascade().detectMultiScale(img, 1.1, 5)
        if len(faces) == 0:
            return result

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cx = (x + w / 2) / W
        cy = (y + h / 2) / H

        placement = subject_placement_score(cx, cy)
        coverage = face_coverage_score(w * h, W * H)
        edge = edge_penalty(x, y, w, h, W, H)
        tilt = tilt_score(img)
        composition = 0.30 * placement + 0.30 * coverage + 0.20 * (1 - edge) + 0.20 * tilt

        result.update(
            {
                "subject_placement": placement,
                "face_coverage": coverage,
                "edge_penalty": edge,
                "tilt_score": tilt,
                "composition_score": composition,
            }
        )
        return result
    except Exception:
        return result


def main():
    df = pd.read_csv(INPUT)
    rows = df[["file_path"] + (["asset_id"] if "asset_id" in df.columns else [])].drop_duplicates(subset=["file_path"]).to_dict("records")

    print("workers =", WORKERS)
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        out_rows = list(
            tqdm(
                executor.map(process_row, rows),
                total=len(rows),
                desc="Computing composition",
                unit="image",
            )
        )

    out = pd.DataFrame(out_rows)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    print("processed =", len(out))
    print("saved_to =", OUT)


if __name__ == "__main__":
    main()
