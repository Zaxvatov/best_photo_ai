from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm

try:
    import config_paths as paths
except ImportError as e:
    raise ImportError("config_paths.py должен быть доступен для импорта") from e

REQUIRED_VARS = [
    "PHOTO_INDEX",
    "SUBJECT",
]

missing = [name for name in REQUIRED_VARS if not hasattr(paths, name)]
if missing:
    available = sorted(name for name in dir(paths) if name.isupper())
    raise ImportError(
        f"config_paths.py не содержит: {', '.join(missing)}. "
        f"Доступные переменные: {available}"
    )

register_heif_opener()

INPUT = Path(paths.PHOTO_INDEX)
OUT = Path(paths.SUBJECT)
WORKERS = max(2, min(8, max(2, (os.cpu_count() or 8) // 2)))
THREAD_LOCAL = threading.local()
MAX_ANALYSIS_DIM = 960

cv2.setNumThreads(1)


def get_cascade() -> cv2.CascadeClassifier:
    cascade = getattr(THREAD_LOCAL, "cascade", None)
    if cascade is None:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        THREAD_LOCAL.cascade = cascade
    return cascade


def compute_subject(path: Path) -> float:
    try:
        img = np.array(Image.open(path).convert("L"))
        h0, w0 = img.shape
        max_dim = max(h0, w0)
        if max_dim > MAX_ANALYSIS_DIM:
            scale = MAX_ANALYSIS_DIM / max_dim
            new_w = max(1, int(w0 * scale))
            new_h = max(1, int(h0 * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape

        faces = get_cascade().detectMultiScale(
            img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        if len(faces) == 0:
            return 0.0

        scores = []
        for (x, y, fw, fh) in faces:
            face_area = (fw * fh) / (w * h)
            cx = x + fw / 2
            cy = y + fh / 2
            center_dist = np.sqrt(((cx - w / 2) / w) ** 2 + ((cy - h / 2) / h) ** 2)
            center_score = max(0.0, 1 - center_dist * 2)
            scores.append(0.6 * face_area + 0.4 * center_score)

        return float(max(scores))
    except Exception:
        return 0.0


def compute_row(row: dict[str, object]) -> dict[str, object]:
    file_path = str(row["file_path"])
    result = {
        "file_path": file_path,
        "subject_score": compute_subject(Path(file_path)),
    }
    if "asset_id" in row:
        result["asset_id"] = row["asset_id"]
    return result


def main() -> None:
    df = pd.read_csv(INPUT)
    rows = df[["file_path"] + (["asset_id"] if "asset_id" in df.columns else [])].drop_duplicates(subset=["file_path"]).to_dict("records")

    print("workers =", WORKERS)
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        out_rows = list(
            tqdm(
                executor.map(compute_row, rows),
                total=len(rows),
                desc="Computing subject",
                unit="img",
            )
        )

    out = pd.DataFrame(out_rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    print("processed =", len(out))
    print("saved_to =", OUT)


if __name__ == "__main__":
    main()
