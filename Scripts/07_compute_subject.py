import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm

from config_paths import SIMILAR_GROUPS_CSV, SUBJECT_SCORES_CSV

register_heif_opener()

INPUT = SIMILAR_GROUPS_CSV
OUT = SUBJECT_SCORES_CSV

CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def compute_subject(path: Path) -> float:
    try:
        img = np.array(Image.open(path).convert("L"))
        h, w = img.shape

        faces = CASCADE.detectMultiScale(
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

            center_dist = np.sqrt(
                ((cx - w / 2) / w) ** 2
                + ((cy - h / 2) / h) ** 2
            )

            center_score = max(0.0, 1 - center_dist * 2)
            score = 0.6 * face_area + 0.4 * center_score
            scores.append(score)

        return float(max(scores))

    except Exception:
        return 0.0


def main() -> None:
    df = pd.read_csv(INPUT)

    rows = []
    for p in tqdm(df["file_path"], total=len(df), desc="Computing subject", unit="img"):
        rows.append((p, compute_subject(Path(p))))

    out = pd.DataFrame(rows, columns=[
        "file_path",
        "subject_score",
    ])

    out.to_csv(OUT, index=False)

    print("processed =", len(out))
    print("saved_to =", OUT)


if __name__ == "__main__":
    main()
