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
from send2trash import send2trash
import io

register_heif_opener()

DATA = r"D:\\photo_ai\\data\\index\\review_groups.csv"
CURATED = Path(r"D:\\photo_ai\\library_curated")

st.set_page_config(layout="wide")

# global compact style
st.markdown(
    """
    <style>
    

    .block-container {
        padding-top: 0.3rem;
        padding-bottom: 0rem;
    }
    h1 {
        margin-top: 0rem;
        margin-bottom: 0rem;
        font-size: 1.55rem;
        line-height: 1.05;
    }
    .metric-line {
        line-height: 1;
        margin: 0;
        font-size: 0.85rem;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        padding: 0.15rem 0.55rem;
        min-height: 2rem;
        font-size: 0.9rem;
        margin-top: 0.95rem;
    }
    div[data-testid="stCheckbox"] {
        margin-bottom: -0.5rem;
        margin-top: 0.95rem;
        white-space: nowrap;
    }
    div[data-testid="stImage"] img {
        margin-top: 0.1rem;
    }
    div[data-testid="stHorizontalBlock"] {
        align-items: start;
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

# keep visually identical files next to each other
# first group by filename, then keep BEST before the duplicate
if not g.empty:
    g["file_name"] = g["file_path"].astype(str).map(lambda p: Path(p).name.lower())
    g["file_stem"] = g["file_path"].astype(str).map(lambda p: Path(p).stem.lower())
    if "is_best" in g.columns:
        g["best_sort"] = g["is_best"].apply(lambda x: 0 if x else 1)
        g = g.sort_values(["file_stem", "file_name", "best_sort", "file_path"])
        g = g.drop(columns=["best_sort"])
    else:
        g = g.sort_values(["file_stem", "file_name", "file_path"])

selected_in_group = []

st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
head_col1, head_col2, head_col3 = st.columns([2.0, 3.0, 1.0])
with head_col1:
    st.markdown(
        f"<div style='font-size:2.2rem;font-weight:700;line-height:1.1;white-space:nowrap'>Group {gid}</div>",
        unsafe_allow_html=True,
    )
with head_col2:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    show_all = st.checkbox("Показывать все метрики", value=False, key=f"show_all_{gid}")
with head_col3:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    delete_clicked = st.button(
        "Удалить",
        type="primary",
        use_container_width=True,
        key=f"delete_{gid}",
    )

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

    if metric in {"best_file", "json_path"}:
        return shorten_photo_path(value)

    if metric == "file_size":
        try:
            b = float(value)
            if b < 1024:
                return f"{int(b)} B"
            if b < 1024**2:
                return f"{b/1024:.1f} KB"
            return f"{b/1024**2:.2f} MB"
        except Exception:
            return value

    if metric == "pixels":
        try:
            return f"{int(value):,} px"
        except Exception:
            return value

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


def trash_path(value):
    if pd.isna(value) or value in (None, "", 0, "0"):
        return False, None, "empty"
    p = Path(str(value))
    if not p.exists():
        return False, p, "missing"
    send2trash(str(p))
    return True, p, "trashed"


if delete_clicked:
    selected_rows = []
    for _, row in g.iterrows():
        raw_path = str(row.get("file_path", ""))
        key = f"select::{gid}::{raw_path}"
        if st.session_state.get(key, False):
            selected_rows.append(row)

    if not selected_rows:
        st.warning("Ничего не выбрано")
    else:
        removed_paths = []
        deleted_files = []
        deleted_json = []
        missing_files = []
        missing_json = []

        for _, row in pd.DataFrame(selected_rows).iterrows():
            file_path = row.get("file_path", "")
            ok_file, p_file, status_file = trash_path(file_path)
            if ok_file:
                deleted_files.append(str(p_file))
            elif status_file == "missing" and p_file is not None:
                missing_files.append(str(p_file))

            json_path = row.get("json_path", None)
            ok_json, p_json, status_json = trash_path(json_path)
            if ok_json:
                deleted_json.append(str(p_json))
            elif status_json == "missing" and p_json is not None:
                missing_json.append(str(p_json))

            removed_paths.append(str(file_path))

        df = df[~df["file_path"].astype(str).isin(removed_paths)].copy()
        df.to_csv(DATA, index=False)

        for file_path in removed_paths:
            st.session_state.pop(f"select::{gid}::{file_path}", None)

        st.success(
            f"Удалено фото: {len(deleted_files)}; json: {len(deleted_json)}; строк из review CSV: {len(removed_paths)}"
        )
        if missing_files or missing_json:
            st.info(
                f"Не найдены при удалении — фото: {len(missing_files)}, json: {len(missing_json)}"
            )

        updated_groups = sorted(df["group_id"].unique()) if not df.empty else []
        if updated_groups:
            if gid in updated_groups:
                current_pos = updated_groups.index(gid)
                next_pos = min(current_pos + 1, len(updated_groups) - 1)
            else:
                old_pos = groups.index(gid) if gid in groups else 0
                next_pos = min(old_pos, len(updated_groups) - 1)
            st.session_state.group_idx = next_pos
        else:
            st.session_state.group_idx = 0

        st.rerun()


visible_metric_cols = metric_cols if show_all else [
    c for c in metric_cols
    if metric_values_differ(g[c])
]

cols = st.columns(len(g))

for i, (_, row) in enumerate(g.iterrows()):

    raw_path = Path(row.get("file_path", ""))
    path = resolve_path(row.get("file_path", ""))

    with cols[i]:

        # checkbox ABOVE the image
        select_key = f"select::{gid}::{row.get('file_path','')}"
        is_selected = st.checkbox(
            "select",
            key=select_key,
            label_visibility="collapsed",
        )

        if is_selected:
            selected_in_group.append(str(row.get("file_path", "")))

        # image preview
        if path.exists():
            try:
                img = Image.open(path)
                img = ImageOps.exif_transpose(img)
                st.image(img, use_container_width=True)
            except Exception:
                st.error("Preview error")
        else:
            st.error("File not found")

        name = path.name
        label = "BEST" if bool(row.get("is_best")) else "FILE"
        st.markdown(f"**{label}: {name}**")

        # metrics background
        metrics_bg = "#dff5df" if bool(row.get("is_best")) else "transparent"

        st.markdown(
            f"<div style='background:{metrics_bg};padding:6px;border-radius:6px'>",
            unsafe_allow_html=True,
        )

        for metric in visible_metric_cols:
            val = format_metric_value(metric, row.get(metric))

            # highlight json_path = 0
            if metric == "json_path" and (str(row.get(metric)) in ["0", "0.0"]):
                st.markdown(
                    f"<p class='metric-line' style='background:#ffe6ea;padding:2px 4px;border-radius:4px'><b>{metric}</b>: {val}</p>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<p class='metric-line'><b>{metric}</b>: {val}</p>",
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)
