import torch
import open_clip
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

INPUT = r"D:\photo_ai\data\index\similar_groups.csv"
OUT = r"D:\photo_ai\data\index\aesthetic_scores.csv"

MODEL_PATH = r"D:\photo_ai\models\sa_0_4_vit_l_14_linear.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="openai"
)

model = model.to(device)
model.eval()

predictor = torch.nn.Linear(768, 1)
predictor.load_state_dict(torch.load(MODEL_PATH))
predictor = predictor.to(device)
predictor.eval()

df = pd.read_csv(INPUT)

rows = []

for p in df.file_path:

    path = Path(p)

    try:

        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model.encode_image(image)
            features = features / features.norm(dim=-1, keepdim=True)
            score = predictor(features).item()

        rows.append((p, score))

    except:
        rows.append((p, 0))

out = pd.DataFrame(rows, columns=["file_path", "aesthetic_score"])

out.to_csv(OUT, index=False)

print("processed =", len(out))
print("saved_to =", OUT)