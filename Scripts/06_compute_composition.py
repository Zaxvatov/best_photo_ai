import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

INPUT = r"D:\photo_ai\data\index\similar_groups.csv"
OUT = r"D:\photo_ai\data\index\composition_scores.csv"

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def subject_placement_score(cx, cy):
    thirds = [(1/3,1/3),(2/3,1/3),(1/3,2/3),(2/3,2/3)]
    d = min(np.sqrt((cx-x)**2 + (cy-y)**2) for x,y in thirds)
    return max(0, 1 - d*1.5)

def face_coverage_score(face_area, img_area):
    r = face_area / img_area
    if 0.05 <= r <= 0.35:
        return 1
    if r < 0.05:
        return r / 0.05
    return max(0, 1 - (r-0.35))

def edge_penalty(x,y,w,h,W,H):
    margin = 0.05
    if x < W*margin or y < H*margin:
        return 1
    if x+w > W*(1-margin) or y+h > H*(1-margin):
        return 1
    return 0

def tilt_score(img):
    edges = cv2.Canny(img,50,150)
    lines = cv2.HoughLines(edges,1,np.pi/180,150)
    if lines is None:
        return 1
    angles = []
    for rho,theta in lines[:,0]:
        angle = abs(theta - np.pi/2)
        angles.append(angle)
    mean_angle = np.mean(angles)
    return max(0,1-mean_angle*2)

df = pd.read_csv(INPUT)

rows = []

for p in df.file_path:

    path = Path(p)

    try:
        img = np.array(Image.open(path).convert("L"))

        H,W = img.shape

        faces = cascade.detectMultiScale(img,1.1,5)

        if len(faces)==0:
            rows.append((p,0,0,0,0,0))
            continue

        x,y,w,h = max(faces, key=lambda f:f[2]*f[3])

        cx = (x+w/2)/W
        cy = (y+h/2)/H

        placement = subject_placement_score(cx,cy)

        coverage = face_coverage_score(w*h, W*H)

        edge = edge_penalty(x,y,w,h,W,H)

        tilt = tilt_score(img)

        composition = (
            0.30*placement +
            0.30*coverage +
            0.20*(1-edge) +
            0.20*tilt
        )

        rows.append((p,placement,coverage,edge,tilt,composition))

    except:
        rows.append((p,0,0,0,0,0))

out = pd.DataFrame(rows, columns=[
    "file_path",
    "subject_placement",
    "face_coverage",
    "edge_penalty",
    "tilt_score",
    "composition_score"
])

out.to_csv(OUT,index=False)

print("processed =",len(out))
print("saved_to =",OUT)