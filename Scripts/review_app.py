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
from html import escape
import io
import json

try:
    import config_paths as cfg
except ImportError as e:
    raise SystemExit(
        "config_paths.py is required for review_app.py to resolve review and curated paths."
    ) from e

register_heif_opener()

DATA = str(Path(cfg.REVIEW_GROUPS))
CURATED = Path(cfg.CURATED_LIBRARY_DIR)
PHOTO_FEATURES = Path(cfg.PHOTO_FEATURES) if hasattr(cfg, "PHOTO_FEATURES") else None
PHOTO_SEMANTIC = Path(cfg.PHOTO_SEMANTIC_SCORES) if hasattr(cfg, "PHOTO_SEMANTIC_SCORES") else None

st.set_page_config(layout="wide")

# global compact style
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: unset !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        min-width: unset !important;
        width: auto !important;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    .st-key-sidebar_header {
        position: sticky;
        top: 0;
        z-index: 1001;
        background: #f0f2f6;
        padding-top: 0.2rem;
        padding-bottom: 0.55rem;
        border-bottom: 1px solid rgba(49, 51, 63, 0.24);
    }
    .st-key-sidebar_scene_list {
        max-height: calc(100vh - 150px);
        overflow-y: auto;
        padding-top: 0.45rem;
    }
    .st-key-sidebar_scene_list div[data-testid="stRadio"] {
        padding-right: 0.2rem;
    }
    .st-key-sidebar_scene_list div[data-testid="stRadio"] label p {
        font-family: Consolas, "Courier New", monospace;
        font-size: 0.95rem;
    }
    .st-key-main_header {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: rgba(255, 255, 255, 0.98);
        padding-top: 0.85rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(49, 51, 63, 0.12);
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    h1 {
        margin-top: 0rem;
        margin-bottom: 0rem;
        font-size: 1.85rem;
        line-height: 1.05;
    }
    .scene-title {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.15;
        padding-top: 0.35rem;
        margin-top: 0;
        white-space: nowrap;
    }
    .metric-line {
        line-height: 1;
        margin: 0;
        font-size: 0.85rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
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
    .metrics-label {
        font-weight: 700;
        line-height: 1.15;
        font-size: 0.85rem;
        margin-bottom: 0.7rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metrics-value {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        line-height: 1.15;
        font-size: 0.85rem;
        margin-bottom: 0.7rem;
    }
    .metrics-value.missing-json {
        background: #ffe6ea;
        padding: 2px 4px;
        border-radius: 4px;
    }
    .sidebar-title {
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 0.45rem;
    }
    .scene-list-header {
        display: grid;
        grid-template-columns: 24px 1fr 1fr;
        gap: 0.45rem;
        align-items: center;
        font-size: 0.84rem;
        font-weight: 700;
        color: #31333f;
        margin-bottom: 0.2rem;
        padding-right: 0.4rem;
    }
    .scene-list-header span {
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not Path(DATA).exists():
    st.error(f"Review file not found: {DATA}")
    st.stop()

df = pd.read_csv(DATA)


def merge_optional_artifact(base_df: pd.DataFrame, artifact_path: Path | None, desired_columns: list[str]) -> pd.DataFrame:
    if artifact_path is None or not artifact_path.exists():
        return base_df

    missing_columns = [column for column in desired_columns if column not in base_df.columns]
    if not missing_columns:
        return base_df

    artifact_df = pd.read_csv(artifact_path)
    merge_columns = [column for column in desired_columns if column in artifact_df.columns and column not in base_df.columns]
    if not merge_columns:
        return base_df

    if "asset_id" not in base_df.columns or "asset_id" not in artifact_df.columns:
        raise KeyError("Review layer expects asset_id in normalized artifacts.")

    artifact_df = artifact_df[["asset_id", *merge_columns]].drop_duplicates(subset=["asset_id"])
    return base_df.merge(artifact_df, on="asset_id", how="left")


df = merge_optional_artifact(
    df,
    PHOTO_FEATURES,
    [
        "primary_file_path",
        "json_path",
        "album_path",
        "sidecar_paths",
        "sidecar_count",
        "has_sidecar",
        "width",
        "height",
        "file_size",
        "content_type_file",
        "sharpness",
        "subject_placement",
        "face_coverage",
        "edge_penalty",
        "tilt_score",
        "composition_score",
        "subject_score",
    ],
)
df = merge_optional_artifact(
    df,
    PHOTO_SEMANTIC,
    [
        "content_type_group",
        "content_type_scene",
        "aesthetic_score",
        "scene_group_id",
        "group_id",
    ],
)

if "group_id" not in df.columns:
    st.error("Column 'group_id' not found in review file.")
    st.stop()

if "scene_group_id" not in df.columns:
    df["scene_group_id"] = df["group_id"]

if "merge_scene_mode" not in st.session_state:
    st.session_state.merge_scene_mode = True
if "filter_people" not in st.session_state:
    st.session_state.filter_people = True
if "filter_landscapes" not in st.session_state:
    st.session_state.filter_landscapes = True
if "filter_documents" not in st.session_state:
    st.session_state.filter_documents = True

merge_scene_mode = bool(st.session_state.merge_scene_mode)
previous_merge_scene_mode = st.session_state.get("previous_merge_scene_mode", merge_scene_mode)


def normalize_content_label(value: object) -> str:
    raw = str(value).strip().lower()
    mapping = {
        "people": "people",
        "landscape": "landscapes",
        "landscapes": "landscapes",
        "document": "documents",
        "documents": "documents",
    }
    return mapping.get(raw, "")


def tags_to_display(value: object) -> str:
    tags = [normalize_content_label(part) for part in str(value).split(",")]
    tags = [tag for tag in tags if tag]
    labels = {
        "people": "Люди",
        "landscapes": "Пейзажи",
        "documents": "Документы",
    }
    return ", ".join(labels[tag] for tag in tags)


def pick_preferred_strict_group(scene_rows: pd.DataFrame) -> int:
    strict_groups = sorted(scene_rows["group_id"].dropna().astype(int).unique())
    if not strict_groups:
        return 0

    remembered = st.session_state.get("last_strict_group_id")
    if remembered in strict_groups:
        return int(remembered)

    if "is_best" in scene_rows.columns:
        best_rows = scene_rows[scene_rows["is_best"] == True].copy()
        if not best_rows.empty:
            if "final_score" in best_rows.columns:
                best_rows = best_rows.sort_values("final_score", ascending=False)
            return int(best_rows["group_id"].iloc[0])

    return int(strict_groups[0])


if "current_group_id" not in st.session_state:
    st.session_state.current_group_id = None

if st.session_state.current_group_id is not None and previous_merge_scene_mode != merge_scene_mode:
    current_gid = int(st.session_state.current_group_id)
    if previous_merge_scene_mode and not merge_scene_mode:
        scene_rows = df[df["scene_group_id"].fillna(df["group_id"]).astype(int) == current_gid]
        if not scene_rows.empty:
            st.session_state.current_group_id = pick_preferred_strict_group(scene_rows)
    elif not previous_merge_scene_mode and merge_scene_mode:
        strict_rows = df[df["group_id"].astype(int) == current_gid]
        if not strict_rows.empty:
            mapped_scene_gid = int(strict_rows["scene_group_id"].fillna(strict_rows["group_id"]).iloc[0])
            st.session_state.current_group_id = mapped_scene_gid

display_group_col = "scene_group_id" if merge_scene_mode else "group_id"
display_group_label = "Сцена"
df["_display_group_id"] = df[display_group_col].fillna(df["group_id"]).astype(int)
if "content_type_file" not in df.columns:
    df["content_type_file"] = ""
if "content_type_group" not in df.columns:
    df["content_type_group"] = df["content_type_file"]
if "content_type_scene" not in df.columns:
    df["content_type_scene"] = df["content_type_group"]

scene_categories: dict[int, set[str]] = {}
for display_group_id, scene_rows in df.groupby("_display_group_id"):
    scene_tags: set[str] = set()
    for col in ("content_type_scene", "content_type_group", "content_type_file"):
        if col in scene_rows.columns:
            for value in scene_rows[col].tolist():
                for part in str(value).split(","):
                    normalized = normalize_content_label(part)
                    if normalized:
                        scene_tags.add(normalized)
    scene_categories[int(display_group_id)] = scene_tags

enabled_categories: set[str] = set()
if st.session_state.filter_people:
    enabled_categories.add("people")
if st.session_state.filter_landscapes:
    enabled_categories.add("landscapes")
if st.session_state.filter_documents:
    enabled_categories.add("documents")

groups = sorted(
    group_id
    for group_id in df["_display_group_id"].unique()
    if scene_categories.get(int(group_id), set()) & enabled_categories
)

if not groups:
    st.warning("Нет сцен для выбранных фильтров.")
    st.stop()

if "current_group_id" not in st.session_state or st.session_state.current_group_id not in groups:
    st.session_state.current_group_id = groups[0]


def format_scene_option(group_value: int) -> str:
    rows = df[df["_display_group_id"] == group_value]
    if rows.empty:
        return str(group_value)

    strict_groups = sorted(rows["group_id"].dropna().astype(int).unique())
    merged_group = int(rows["scene_group_id"].fillna(rows["group_id"]).iloc[0])

    if merge_scene_mode:
        strict_label = ", ".join(str(item) for item in strict_groups)
        if strict_label == str(group_value):
            strict_label = ""
        return f"{group_value:<5} {strict_label}"

    strict_value = int(group_value)
    if merged_group == strict_value:
        return f"{merged_group:<5}"
    return f"{merged_group:<5} {strict_value}"

def move_group(step: int) -> None:
    current_idx = groups.index(st.session_state.current_group_id)
    next_idx = max(0, min(current_idx + step, len(groups) - 1))
    st.session_state.current_group_id = groups[next_idx]
    st.session_state.sidebar_scene = st.session_state.current_group_id

with st.sidebar:
    sidebar_header = st.container(key="sidebar_header")
    with sidebar_header:
        st.markdown("<div class='sidebar-title'>Сцены</div>", unsafe_allow_html=True)
        nav_col1, nav_col2 = st.columns(2)
        if nav_col1.button("↑", use_container_width=True):
            move_group(-1)
        if nav_col2.button("↓", use_container_width=True):
            move_group(1)
        st.markdown(
            """
            <div class='scene-list-header'>
                <span></span>
                <span title='Номер общей сцены с общим фоном и похожей сценой'>Общая</span>
                <span title='Номер частной сцены с тем же фоном, но с другими участниками'>Частная</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if "sidebar_scene" not in st.session_state or st.session_state.sidebar_scene not in groups:
        st.session_state.sidebar_scene = st.session_state.current_group_id

    sidebar_scene_list = st.container(key="sidebar_scene_list")
    with sidebar_scene_list:
        gid = st.radio(
            "Сцены",
            groups,
            key="sidebar_scene",
            label_visibility="collapsed",
            format_func=format_scene_option,
        )
        st.session_state.current_group_id = gid

gid = st.session_state.current_group_id

g = df[df["_display_group_id"] == gid].copy()

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

header_container = st.container(key="main_header")
with header_container:
    head_col1, head_col2, head_col3 = st.columns([1.7, 4.2, 1.1])
    with head_col1:
        title_suffix = ""
        if merge_scene_mode and "group_id" in g.columns:
            strict_count = int(g["group_id"].nunique())
            title_suffix = f" ({strict_count} группы)" if strict_count > 1 else ""
        st.markdown(
            f"<div class='scene-title'>{display_group_label} {gid}{title_suffix}</div>",
            unsafe_allow_html=True,
        )
    with head_col2:
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        toggle_col1, toggle_col2, toggle_col3, toggle_col4, toggle_col5 = st.columns(5)
        with toggle_col1:
            merge_scene_mode = st.checkbox(
                "Объединять похожие сцены",
                value=True,
                key="merge_scene_mode",
                help="Показывать вместе строгие группы, если они входят в одну scene_group_id.",
            )
        with toggle_col2:
            show_all = st.checkbox("Показывать все метрики", value=False, key=f"show_all_{gid}")
        with toggle_col3:
            st.checkbox("Люди", value=True, key="filter_people")
        with toggle_col4:
            st.checkbox("Пейзажи", value=True, key="filter_landscapes")
        with toggle_col5:
            st.checkbox("Документы", value=True, key="filter_documents")
    with head_col3:
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        delete_clicked = st.button(
            "Удалить",
            type="primary",
            use_container_width=True,
            key=f"delete_{gid}",
        )

st.session_state.previous_merge_scene_mode = merge_scene_mode
if not g.empty and "group_id" in g.columns:
    st.session_state.last_strict_group_id = int(g["group_id"].mode().iloc[0])

EXCLUDED_METRICS = {
    "file_name",
    "file_stem",
    "best_file",
    "best_score",
    "scene_group_id",
    "scene_group_size",
    "scene_group_group_count",
    "scene_merge_candidate",
    "_display_group_id",
}

metric_cols = sorted([
    c for c in g.columns
    if c not in {"group_id", "file_path", "is_best"}
    and c not in EXCLUDED_METRICS
])

hidden_when_compact = set()

METRIC_LABELS = {
    "aesthetic_n": "Эстетика (норм.)",
    "aesthetic_score": "Эстетика",
    "best_file": "Лучший файл группы",
    "best_score": "Лучший score группы",
    "composition_score": "Композиция",
    "content_type_file": "Тег файла",
    "content_type_group": "Тег группы",
    "content_type_scene": "Тег сцены",
    "edge_penalty": "Штраф за края",
    "face_coverage": "Покрытие лицом",
    "file_name": "Имя файла",
    "file_size": "Размер файла",
    "file_size_n": "Размер файла (норм.)",
    "file_stem": "Основа имени",
    "height": "Высота",
    "json_path": "JSON путь",
    "pixels": "Пиксели",
    "pixels_n": "Пиксели (норм.)",
    "score_v8": "Итоговый балл",
    "sharpness": "Резкость",
    "sharpness_n": "Резкость (норм.)",
    "subject_n": "Субъект (норм.)",
    "subject_placement": "Положение субъекта",
    "subject_score": "Субъект",
    "tilt_score": "Наклон",
    "width": "Ширина",
    "size_wh": "Размер Ш×В",
}

METRIC_HELP = {
    "aesthetic_n": "Формула: aesthetic_n = norm_by_group(aesthetic_score).\nИнтерпретация: 0 — худший кадр в группе по эстетике, 1 — лучший.",
    "aesthetic_score": "Базовая эстетическая оценка модели. Чем выше, тем кадр обычно визуально приятнее.",
    "best_file": "Формула: файл с максимальным итоговым баллом внутри группы.",
    "best_score": "Формула: max(итоговый балл) внутри группы.",
    "composition_score": "Базовая оценка композиции кадра. Чем выше, тем композиция обычно лучше.",
    "content_type_file": "Классификация отдельного файла из индекса: Люди, Пейзажи или Документы.",
    "content_type_group": "Агрегированный тег строгой группы. Помогает проверить, не смешались ли разные типы контента.",
    "content_type_scene": "Агрегированный тег объединённой сцены. Полезно для контроля качества обучения и фильтрации.",
    "edge_penalty": "Штраф за объект, упирающийся в край кадра. 0 — штрафа нет. Чем выше, тем хуже.",
    "face_coverage": "Формула: доля площади кадра, занятой лицом / лицами.\nКак интерпретировать значения:\n0.0 → лиц нет\n0.01–0.1 → лица есть, но маленькие / далеко\n0.1–0.3 → нормальный портрет / сцена с людьми\n0.3+ → крупные лица, близкий план.",
    "file_name": "Полное имя файла с расширением.",
    "file_size": "Размер файла на диске.",
    "file_size_n": "Формула: file_size_n = norm_by_group(file_size).\nИнтерпретация: 0 — самый маленький файл в группе, 1 — самый большой.",
    "file_stem": "Имя файла без расширения.",
    "json_path": "Путь к JSON-sidecar файлу с метаданными.",
    "pixels": "Формула: pixels = width × height.",
    "pixels_n": "Формула: pixels_n = norm_by_group(pixels).\nИнтерпретация: 0 — минимальное разрешение в группе, 1 — максимальное.",
    "score_v8": "Итоговый балл выбора лучшего кадра.\nФормула: взвешенная сумма нормализованных и базовых метрик из builder-скрипта текущей версии pipeline (v8). Этот UI показывает готовый результат из review_groups.csv.",
    "sharpness": "Базовая оценка резкости кадра. Чем выше, тем кадр обычно резче.",
    "sharpness_n": "Формула: sharpness_n = norm_by_group(sharpness).\nИнтерпретация: 0 — самый мягкий кадр в группе, 1 — самый резкий.",
    "subject_n": "Формула: subject_n = norm_by_group(subject_score).\nИнтерпретация: 0 — худший субъект в группе, 1 — лучший.",
    "subject_placement": "Базовая оценка положения субъекта в кадре. Чем выше, тем расположение обычно удачнее.",
    "subject_score": "Итоговая базовая оценка субъекта / лица в кадре. Обычно выше, если субъект читается лучше.",
    "tilt_score": "Базовая оценка наклона кадра. Ближе к 1 — ровнее, ближе к 0 — сильнее завал.",
    "size_wh": "Формула: Размер Ш×В = width × height, выводится как ширина × высота в пикселях.",
}


def shorten_photo_path(value):
    if pd.isna(value):
        return value
    s = str(value)
    marker = "raw_takeout\\"
    idx = s.lower().find(marker.lower())
    if idx >= 0:
        return s[idx + len(marker):]
    return s


def format_metric_value(metric, value, row=None):
    if metric == "size_wh":
        try:
            w = row.get("width") if row is not None else None
            h = row.get("height") if row is not None else None
            if pd.isna(w) or pd.isna(h):
                return None
            return f"{int(float(w))} × {int(float(h))} px"
        except Exception:
            return None

    if pd.isna(value):
        return value

    if metric in {"best_file", "json_path"}:
        return shorten_photo_path(value)

    if metric in {"content_type_file", "content_type_group", "content_type_scene"}:
        return tags_to_display(value)

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


def metric_is_empty_for_all(series):
    for v in series.tolist():
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s not in {"", "0", "0.0", "None", "nan", "NaN"}:
            return False
    return True


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


def parse_sidecar_paths(value) -> list[str]:
    if pd.isna(value) or value in (None, "", 0, "0"):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    return [str(parsed)]


if delete_clicked:
    selected_rows = []
    for _, row in g.iterrows():
        raw_key = str(row.get("asset_id") or row.get("file_path", ""))
        key = f"select::{gid}::{raw_key}"
        if st.session_state.get(key, False):
            selected_rows.append(row)

    if not selected_rows:
        st.warning("Ничего не выбрано")
    else:
        removed_assets = []
        deleted_files = []
        deleted_sidecars = []
        missing_files = []
        missing_sidecars = []

        for _, row in pd.DataFrame(selected_rows).iterrows():
            file_path = row.get("file_path", "")
            ok_file, p_file, status_file = trash_path(file_path)
            if ok_file:
                deleted_files.append(str(p_file))
            elif status_file == "missing" and p_file is not None:
                missing_files.append(str(p_file))

            sidecar_paths = parse_sidecar_paths(row.get("sidecar_paths", None))
            json_path = row.get("json_path", None)
            if json_path and str(json_path).strip():
                sidecar_paths.append(str(json_path))
            sidecar_paths = list(dict.fromkeys(sidecar_paths))

            for sidecar_path in sidecar_paths:
                ok_sidecar, p_sidecar, status_sidecar = trash_path(sidecar_path)
                if ok_sidecar:
                    deleted_sidecars.append(str(p_sidecar))
                elif status_sidecar == "missing" and p_sidecar is not None:
                    missing_sidecars.append(str(p_sidecar))

            removed_assets.append(str(row.get("asset_id") or file_path))

        removed_assets_set = set(removed_assets)
        df = df[~df.apply(lambda row: str(row.get("asset_id") or row.get("file_path", "")) in removed_assets_set, axis=1)].copy()
        df.to_csv(DATA, index=False)

        for asset_key in removed_assets:
            st.session_state.pop(f"select::{gid}::{asset_key}", None)

        st.success(
            f"Удалено primary: {len(deleted_files)}; sidecar: {len(deleted_sidecars)}; строк из review CSV: {len(removed_assets)}"
        )
        if missing_files or missing_sidecars:
            st.info(
                f"Не найдены при удалении — primary: {len(missing_files)}, sidecar: {len(missing_sidecars)}"
            )

        updated_groups = sorted(df["group_id"].unique()) if not df.empty else []
        if merge_scene_mode:
            updated_groups = sorted(df["scene_group_id"].fillna(df["group_id"]).astype(int).unique()) if not df.empty else []
        if updated_groups:
            if gid in updated_groups:
                current_pos = updated_groups.index(gid)
                next_pos = min(current_pos + 1, len(updated_groups) - 1)
            else:
                old_pos = groups.index(gid) if gid in groups else 0
                next_pos = min(old_pos, len(updated_groups) - 1)
            st.session_state.current_group_id = updated_groups[next_pos]
            st.session_state.sidebar_scene = st.session_state.current_group_id
        else:
            st.session_state.current_group_id = 0
            st.session_state.sidebar_scene = 0

        st.rerun()


metric_cols_render = [c for c in metric_cols if c not in {"width", "height"}]
if "width" in g.columns and "height" in g.columns:
    metric_cols_render.append("size_wh")
metric_cols_render = sorted(metric_cols_render)

visible_metric_cols = metric_cols_render if show_all else [
    c for c in metric_cols_render
    if c not in hidden_when_compact and (
        c == "size_wh" or metric_values_differ(g[c])
    )
]

# table-like header row: blank top-left cell + photo columns
image_cols = st.columns([1.35] + [1] * len(g), gap="small")

with image_cols[0]:
    st.markdown("&nbsp;", unsafe_allow_html=True)

cols = image_cols[1:]

for i, (_, row) in enumerate(g.iterrows()):

    raw_path = Path(row.get("file_path", ""))
    path = resolve_path(row.get("file_path", ""))

    with cols[i]:

        # checkbox ABOVE the image
        select_key = f"select::{gid}::{row.get('asset_id') or row.get('file_path','')}"
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
        is_best = bool(row.get("is_best"))

        if is_best:
            st.markdown(
                f"<div style='font-weight:700;color:#2e7d32'>{escape(name)}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"**{escape(name)}**", unsafe_allow_html=True)

# aligned metrics block: uses the same Streamlit columns as the photo row
metric_display_cols = st.columns([1.35] + [1] * len(g), gap="small")

label_parts = []
value_parts: list[list[str]] = [[] for _ in range(len(g))]

for metric in (metric_cols_render if show_all else visible_metric_cols):
    label = escape(METRIC_LABELS.get(metric, metric))
    help_text = escape(METRIC_HELP.get(metric, ""), quote=True)

    numeric_values = []
    for _, r in g.iterrows():
        v = r.get(metric)
        try:
            numeric_values.append(float(v))
        except Exception:
            numeric_values.append(None)

    max_val = None
    if any(v is not None for v in numeric_values):
        max_val = max([v for v in numeric_values if v is not None])

    label_parts.append(f"<div class='metrics-label' title='{help_text}'>{label}</div>")

    for idx, (_, row) in enumerate(g.iterrows()):
        raw_val = row.get(metric)
        val = format_metric_value(metric, raw_val, row=row)
        if val in [None, "", "None"]:
            val = ""

        value_html = escape(str(val))
        cls = "metrics-value"
        style = ""

        try:
            if max_val is not None and float(raw_val) == max_val:
                style = "color:#2e7d32;"
        except Exception:
            pass

        if metric == "json_path" and str(raw_val) in ["0", "0.0"]:
            cls += " missing-json"

        value_parts[idx].append(
            f"<div class='{cls}' style='{style}' title='{help_text}'>{value_html}</div>"
        )

with metric_display_cols[0]:
    st.markdown("".join(label_parts), unsafe_allow_html=True)

for idx, col in enumerate(metric_display_cols[1:]):
    with col:
        st.markdown("".join(value_parts[idx]), unsafe_allow_html=True)

st.markdown(
    """
    <script>
    const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
    if (sidebar) {
      const selected = sidebar.querySelector('input[type="radio"]:checked');
      if (selected) {
        const row = selected.closest('label');
        if (row) {
          row.scrollIntoView({block: 'nearest'});
        }
      }
    }
    </script>
    """,
    unsafe_allow_html=True,
)
