import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

INPUT = r"D:\photo_ai\data\index\similar_groups.csv"
OUT = r"D:\photo_ai\data\index\faces.csv"

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(CASCADE_PATH)


def detect_faces(path: Path):
    try:
        img = np.array(Image.open(path).convert("L"))
        faces = cascade.detectMultiScale(
            img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        count = len(faces)
        largest = max(
            ((w * h) / (img.shape[0] * img.shape[1]) for (_, _, w, h) in faces),
            default=0.0
        )
        return count, float(largest)
    except Exception:
        return None, None


def main():
    df = pd.read_csv(INPUT)

    rows = []
    for p in df["file_path"]:
        count, largest = detect_faces(Path(p))
        rows.append((p, count, largest))

    out = pd.DataFrame(rows, columns=["file_path", "faces", "largest_face"])
    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    print("processed =", len(out))
    print("saved_to =", OUT)


if __name__ == "__main__":
    main()