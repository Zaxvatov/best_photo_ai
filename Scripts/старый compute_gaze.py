import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

INPUT = r"D:\photo_ai\data\index\similar_groups.csv"
OUT = r"D:\photo_ai\data\index\gaze_scores.csv"

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def frontal_face_score(x, y, w, h, W, H):
    """
    Приближённая оценка:
    - лицо должно быть достаточно крупным
    - не слишком у края
    - не слишком вытянуто/странно детектировано
    """
    cx = (x + w / 2) / W
    cy = (y + h / 2) / H
    face_area = (w * h) / (W * H)

    center_score = 1.0 - min(abs(cx - 0.5) * 1.6, 1.0)
    vertical_score = 1.0 - min(abs(cy - 0.45) * 1.8, 1.0)

    ratio = w / h if h > 0 else 0
    ratio_score = max(0.0, 1.0 - min(abs(ratio - 0.78) * 2.5, 1.0))

    size_score = min(face_area / 0.08, 1.0) if face_area < 0.08 else max(0.0, 1.0 - (face_area - 0.08) * 2.0)

    return (
        0.35 * center_score +
        0.20 * vertical_score +
        0.25 * ratio_score +
        0.20 * size_score
    )

df = pd.read_csv(INPUT)

rows = []

for p in df["file_path"]:
    path = Path(p)

    try:
        img = np.array(Image.open(path).convert("L"))
        H, W = img.shape

        faces = cascade.detectMultiScale(
            img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            rows.append((p, 0, 0, 0, 0, 0))
            continue

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        cx = (x + w / 2) / W
        cy = (y + h / 2) / H
        face_ratio = w / h if h > 0 else 0

        gaze_score = frontal_face_score(x, y, w, h, W, H)

        rows.append((p, cx, cy, face_ratio, (w * h) / (W * H), gaze_score))

    except Exception:
        rows.append((p, 0, 0, 0, 0, 0))

out = pd.DataFrame(rows, columns=[
    "file_path",
    "face_cx",
    "face_cy",
    "face_ratio",
    "face_area_ratio",
    "gaze_score"
])

out.to_csv(OUT, index=False)

print("processed =", len(out))
print("saved_to =", OUT)