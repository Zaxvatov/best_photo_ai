import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
import io

register_heif_opener()

DATA = r"D:\photo_ai\data\index\review_groups.csv"
CURATED = Path(r"D:\photo_ai\library_curated")

st.set_page_config(layout="wide")

df = pd.read_csv(DATA)
groups = sorted(df["group_id"].unique())

st.sidebar.title("Groups")
gid = st.sidebar.radio("Group", groups)

g = df[df["group_id"] == gid].copy()
g["best_sort"] = g["is_best"].apply(lambda x: 0 if x else 1)
g = g.sort_values(["best_sort", "file_path"]).drop(columns=["best_sort"])

st.title(f"Group {gid}")

def resolve_path(p: str) -> Path:
    path = Path(str(p))
    if path.exists():
        return path

    alt = CURATED / path.name
    if alt.exists():
        return alt

    return path

def load_preview(path: Path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    else:
        img = img.copy()

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

cols = st.columns(len(g))

for i, (_, row) in enumerate(g.iterrows()):
    raw_path = Path(row["file_path"])
    path = resolve_path(row["file_path"])

    with cols[i]:
        if path.exists():
            try:
                preview = load_preview(path)
                st.image(preview, use_container_width=True)
            except Exception as e:
                st.error("Preview not supported")
                st.code(str(e))
        else:
            st.error("File not found")
            st.code(str(raw_path))

        name = path.name

        if row["is_best"]:
            st.markdown(
                f"<div style='border:5px solid green;padding:5px;font-weight:bold'>{name}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.write(name)

        st.write("faces:", row.get("faces"))
        st.write("sharpness:", row.get("sharpness"))
        st.write("largest_face:", row.get("largest_face"))