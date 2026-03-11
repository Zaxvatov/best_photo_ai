import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener
from concurrent.futures import ProcessPoolExecutor
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

register_heif_opener()

INPUT = r"D:\photo_ai\data\index\similar_groups.csv"
OUT = r"D:\photo_ai\data\index\landmarks_scores.csv"
MODEL = r"D:\photo_ai\models\face_landmarker_v2.task"

WORKERS = 8

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

LANDMARKER = None

def init_worker():
    global LANDMARKER
    register_heif_opener()
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=VisionRunningMode.IMAGE
    )
    LANDMARKER = vision.FaceLandmarker.create_from_options(options)

def dist(a, b):
    return np.linalg.norm(np.array(a[:2]) - np.array(b[:2]))

def eye_open_score(pts, top1, top2, bottom1, bottom2, left, right):
    top = dist(pts[top1], pts[bottom1])
    bottom = dist(pts[top2], pts[bottom2])
    width = dist(pts[left], pts[right])
    if width == 0:
        return 0
    return (top + bottom) / (2 * width)

def iris_center(pts, ids):
    arr = np.array([pts[i][:2] for i in ids])
    return arr.mean(axis=0)

def gaze_eye_score(pts, iris_ids, left_corner, right_corner, top_id, bottom_id):
    iris = iris_center(pts, iris_ids)

    left = np.array(pts[left_corner][:2])
    right = np.array(pts[right_corner][:2])
    top = np.array(pts[top_id][:2])
    bottom = np.array(pts[bottom_id][:2])

    eye_center = (left + right + top + bottom) / 4

    eye_width = np.linalg.norm(right - left)
    eye_height = np.linalg.norm(bottom - top)

    if eye_width == 0 or eye_height == 0:
        return 0

    dx = abs(iris[0] - eye_center[0]) / (eye_width / 2)
    dy = abs(iris[1] - eye_center[1]) / (eye_height / 2)

    score = 1 - min(1, 0.65 * dx + 0.35 * dy)
    return max(0, min(1, score))

def smile_score(pts):
    left_mouth = pts[61]
    right_mouth = pts[291]
    upper_lip = pts[13]
    lower_lip = pts[14]
    left_eye = pts[33]
    right_eye = pts[263]

    mouth_width = dist(left_mouth, right_mouth)
    mouth_open = dist(upper_lip, lower_lip)
    eye_width = dist(left_eye, right_eye)

    if eye_width == 0:
        return 0

    width_ratio = mouth_width / eye_width
    open_ratio = mouth_open / eye_width

    width_score = max(0, min(1, (width_ratio - 0.35) / 0.25))
    open_score = max(0, 1 - abs(open_ratio - 0.05) / 0.06)

    return max(0, min(1, 0.7 * width_score + 0.3 * open_score))

def process_one(p):
    global LANDMARKER
    path = Path(p)

    try:
        img = np.array(Image.open(path).convert("RGB"))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        result = LANDMARKER.detect(mp_image)

        if not result.face_landmarks:
            return (p, 0, 0, 0, 0, 0, 0)

        lm = result.face_landmarks[0]
        pts = [(l.x, l.y, l.z) for l in lm]

        left_eye = eye_open_score(pts, 159, 158, 145, 144, 33, 133)
        right_eye = eye_open_score(pts, 386, 385, 374, 373, 362, 263)
        eyes_open = (left_eye + right_eye) / 2

        left = pts[234][0]
        right = pts[454][0]
        nose = pts[1][0]

        yaw = abs(nose - (left + right) / 2)
        chin = pts[152][1]
        pitch = abs(pts[1][1] - chin)
        symmetry = max(0, 1 - abs(nose - (left + right) / 2) * 4)

        left_gaze = gaze_eye_score(pts, [468, 469, 470, 471, 472], 33, 133, 159, 145)
        right_gaze = gaze_eye_score(pts, [473, 474, 475, 476, 477], 362, 263, 386, 374)
        gaze_to_camera = (left_gaze + right_gaze) / 2

        smile = smile_score(pts)

        return (p, eyes_open, yaw, pitch, symmetry, gaze_to_camera, smile)

    except Exception:
        return (p, 0, 0, 0, 0, 0, 0)

def main():
    df = pd.read_csv(INPUT)
    files = df["file_path"].tolist()

    with ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker) as ex:
        rows = list(ex.map(process_one, files, chunksize=8))

    out = pd.DataFrame(rows, columns=[
        "file_path",
        "eyes_open_score",
        "head_yaw",
        "head_pitch",
        "face_symmetry",
        "gaze_to_camera_score",
        "smile_score"
    ])

    out.to_csv(OUT, index=False)

    print("workers =", WORKERS)
    print("processed =", len(out))
    print("saved_to =", OUT)

if __name__ == "__main__":
    main()