# Review UI for inspecting photo groups
# Requires: streamlit, pandas, pillow, pillow-heif
# Run with:
#   streamlit run review_app.py

try:
    import streamlit as st
except ModuleNotFoundError:
    raise SystemExit(
        "Streamlit is not installed. Install it with: pip install streamlit"
    )

import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
import io

register_heif_opener()

DATA = r"D:\\photo_ai\\data\\index\\review_groups.csv"
CURATED = Path(r"D:\\photo_ai\\library_curated")

st.set_page_config(layout="wide")

# global compact style
st.markdown(
    """
    <style>
    .metric-line {
        line-height: 1;
        margin: 0;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not Path(DATA).exists():
    st.error(f"Review file not found: {DATA}")
    st.stop()

df = pd.read_csv(DATA)

if "group_id" not in df.columns:
    st.error("Column 'group_id' not found in review file.")
    st.stop()

groups = sorted(df["group_id"].unique())

if "group_idx" not in st.session_state:
    st.session_state.group_idx = 0

st.session_state.group_idx = max(0, min(st.session_state.group_idx, len(groups) - 1))

st.sidebar.title("Groups")
nav_col1, nav_col2 = st.sidebar.columns(2)

if nav_col1.button("↑ Up", use_container_width=True):
    st.session_state.group_idx = max(0, st.session_state.group_idx - 1)

if nav_col2.button("↓ Down", use_container_width=True):
    st.session_state.group_idx = min(len(groups) - 1, st.session_state.group_idx + 1)

gid = st.sidebar.radio(
    "Group",
    groups,
    index=st.session_state.group_idx,
)
st.session_state.group_idx = groups.index(gid)

g = df[df["group_id"] == gid].copy()

if "is_best" in g.columns:
    g["best_sort"] = g["is_best"].apply(lambda x: 0 if x else 1)
    g = g.sort_values(["best_sort", "file_path"]).drop(columns=["best_sort"])

st.title(f"Group {gid}")

show_all = st.checkbox("Показывать полные данные / Show full data", value=False)

metric_cols = [
    c for c in g.columns
    if c not in {"group_id", "file_path", "is_best"}
]


def shorten_photo_path(value):
    if pd.isna(value):
        return value
    s = str(value)
    marker = "raw_takeout\\"
    idx = s.lower().find(marker.lower())
    if idx >= 0:
        return s[idx + len(marker):]
    return s


def format_metric_value(metric, value):
    if pd.isna(value):
        return value
    if metric == "best_file":
        return shorten_photo_path(value)
    return value


def metric_values_differ(series):
    normalized = []
    for v in series.tolist():
        if pd.isna(v):
            normalized.append("__NA__")
        else:
            normalized.append(str(v))
    return len(set(normalized)) > 1


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

visible_metric_cols = metric_cols if show_all else [
    c for c in metric_cols
    if metric_values_differ(g[c])
]

cols = st.columns(len(g))

for i, (_, row) in enumerate(g.iterrows()):
    raw_path = Path(row.get("file_path", ""))
    path = resolve_path(row.get("file_path", ""))

    with cols[i]:
        card_bg = "#dff5df" if bool(row.get("is_best")) else "#f7f7f7"
        st.markdown(
            f"""
            <div style='background:{card_bg};padding:12px;border-radius:12px;margin-bottom:12px;'>
            """,
            unsafe_allow_html=True,
        )

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
        label = "BEST" if bool(row.get("is_best")) else "FILE"
        st.markdown(f"**{label}: {name}**")

        metrics_bg = "#dff5df" if bool(row.get("is_best")) else "transparent"

        st.markdown(f"<div style='background:{metrics_bg};padding:6px;border-radius:6px'>", unsafe_allow_html=True)

        for metric in visible_metric_cols:
            val = format_metric_value(metric, row.get(metric))
            st.markdown(f"<p class='metric-line'><b>{metric}</b>: {val}</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
