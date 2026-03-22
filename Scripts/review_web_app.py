from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from send2trash import send2trash

import config_paths as cfg


SCRIPT_DIR = Path(__file__).resolve().parent
STATIC_DIR = SCRIPT_DIR.parent / "viewer_static"
PHOTO_DATA = Path(cfg.REVIEW_GROUPS)
VIDEO_DATA = Path(cfg.VIDEO_REVIEW_GROUPS)


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

METRIC_LABELS = {
    "aesthetic_n": "Эстетика (норм.)",
    "aesthetic_score": "Эстетика",
    "audio_codec": "Аудио кодек",
    "audio_present_final": "Есть аудио",
    "best_file": "Лучший файл группы",
    "best_score": "Лучший score группы",
    "bitrate_kbps_final": "Битрейт",
    "composition_score": "Композиция",
    "content_type_file": "Тег файла",
    "content_type_group": "Тег группы",
    "content_type_scene": "Тег сцены",
    "duration_sec_final": "Длительность",
    "edge_penalty": "Штраф за края",
    "face_coverage": "Покрытие лицом",
    "file_name": "Имя файла",
    "file_size": "Размер файла",
    "file_size_n": "Размер файла (норм.)",
    "final_score": "Итоговый балл",
    "fps_final": "FPS",
    "fps_norm": "FPS (норм.)",
    "has_live_photo_pair": "Live Photo пара",
    "json_path": "JSON путь",
    "live_photo_penalty": "Штраф Live Photo",
    "live_photo_photo_file_name": "Фото Live Photo",
    "live_photo_photo_primary_file_path": "Путь фото Live Photo",
    "live_photo_status": "Статус Live Photo",
    "pixels": "Пиксели",
    "pixels_n": "Пиксели (норм.)",
    "pixels_norm": "Пиксели (норм.)",
    "sharpness": "Резкость",
    "sharpness_n": "Резкость (норм.)",
    "sidecar_norm": "Sidecar (норм.)",
    "subject_n": "Субъект (норм.)",
    "subject_placement": "Положение субъекта",
    "subject_score": "Субъект",
    "tilt_score": "Наклон",
    "video_codec": "Видео кодек",
    "video_group_id": "Видео группа",
    "video_group_size": "Размер видео группы",
    "video_metrics_backend": "Backend метрик",
    "video_metrics_status": "Статус метрик",
    "video_score": "Video score",
    "video_score_raw": "Video score без штрафа",
    "width_final": "Ширина",
    "height_final": "Высота",
    "size_wh": "Размер Ш×В",
}

METRIC_HELP = {
    "aesthetic_n": "Нормализованная эстетика внутри группы.",
    "aesthetic_score": "Базовая эстетическая оценка модели.",
    "best_file": "Файл с максимальным итоговым баллом внутри группы.",
    "best_score": "Максимальный итоговый балл внутри группы.",
    "bitrate_kbps_final": "Итоговый битрейт ролика в kbps.",
    "composition_score": "Базовая оценка композиции.",
    "content_type_file": "Классификация отдельного файла.",
    "content_type_group": "Агрегированный тег строгой группы.",
    "content_type_scene": "Агрегированный тег объединённой сцены.",
    "duration_sec_final": "Итоговая длительность ролика или кадра.",
    "edge_penalty": "Штраф за объект у края кадра.",
    "face_coverage": "Доля площади кадра, занятой лицом или лицами.",
    "file_name": "Имя файла с расширением.",
    "file_size": "Размер файла на диске.",
    "file_size_n": "Нормализованный размер файла внутри группы.",
    "final_score": "Итоговый балл для выбора лучшего элемента.",
    "fps_final": "Частота кадров видео.",
    "fps_norm": "Нормализованный FPS внутри video-группы.",
    "has_live_photo_pair": "Есть ли связанная Live Photo фотография.",
    "json_path": "Путь к JSON sidecar.",
    "live_photo_penalty": "Штраф, применённый к live-photo ролику в video ranking.",
    "live_photo_photo_file_name": "Имя связанного фото из Live Photo пары.",
    "live_photo_photo_primary_file_path": "Путь к связанному фото из Live Photo пары.",
    "live_photo_status": "Сила совпадения Live Photo пары.",
    "pixels": "Количество пикселей кадра.",
    "pixels_n": "Нормализованное разрешение внутри группы.",
    "pixels_norm": "Нормализованное разрешение внутри video-группы.",
    "sharpness": "Базовая оценка резкости.",
    "sharpness_n": "Нормализованная резкость внутри группы.",
    "sidecar_norm": "Нормализованное число sidecar-файлов внутри video-группы.",
    "subject_n": "Нормализованная оценка субъекта внутри группы.",
    "subject_placement": "Оценка положения субъекта в кадре.",
    "subject_score": "Оценка качества субъекта.",
    "tilt_score": "Оценка завала горизонта.",
    "video_codec": "Видео кодек файла.",
    "video_group_id": "Идентификатор video-группы.",
    "video_group_size": "Размер video-группы.",
    "video_metrics_backend": "Backend, которым собраны техметрики видео.",
    "video_metrics_status": "Статус расчёта видео-метрик.",
    "video_score": "Итоговый ranking score для видео.",
    "video_score_raw": "Video score до применения live-photo штрафа.",
    "width_final": "Итоговая ширина видео.",
    "height_final": "Итоговая высота видео.",
    "size_wh": "Размер в формате ширина × высота.",
}


app = FastAPI(title="Best Photo AI Review")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DeleteRequest(BaseModel):
    media_mode: str
    asset_ids: list[str]


def normalize_content_label(value: object) -> str:
    raw = str(value).strip().lower()
    mapping = {
        "people": "people",
        "landscape": "landscapes",
        "landscapes": "landscapes",
        "document": "documents",
        "documents": "documents",
        "video": "video",
    }
    return mapping.get(raw, "")


def tags_to_display(value: object) -> str:
    tags = [normalize_content_label(part) for part in str(value).split(",")]
    tags = [tag for tag in tags if tag]
    labels = {
        "people": "Люди",
        "landscapes": "Пейзажи",
        "documents": "Документы",
        "video": "Видео",
    }
    return ", ".join(labels[tag] for tag in tags)


def parse_sidecar_paths(value: object) -> list[str]:
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


def resolve_data_path(media_mode: str) -> Path:
    return PHOTO_DATA if media_mode == "photo" else VIDEO_DATA


def load_df(media_mode: str) -> pd.DataFrame:
    path = resolve_data_path(media_mode)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Review file not found: {path}")
    df = pd.read_csv(path)
    if "group_id" not in df.columns:
        raise HTTPException(status_code=500, detail="Column 'group_id' not found in review file.")
    if "scene_group_id" not in df.columns:
        df["scene_group_id"] = df["group_id"]
    if "content_type_file" not in df.columns:
        df["content_type_file"] = ""
    if "content_type_group" not in df.columns:
        df["content_type_group"] = df["content_type_file"]
    if "content_type_scene" not in df.columns:
        df["content_type_scene"] = df["content_type_group"]
    return df


def scene_categories(df: pd.DataFrame, display_group_col: str) -> dict[int, set[str]]:
    categories: dict[int, set[str]] = {}
    temp = df.copy()
    temp["_display_group_id"] = temp[display_group_col].fillna(temp["group_id"]).astype(int)
    for display_group_id, scene_rows in temp.groupby("_display_group_id"):
        scene_tags: set[str] = set()
        for col in ("content_type_scene", "content_type_group", "content_type_file"):
            if col in scene_rows.columns:
                for value in scene_rows[col].tolist():
                    for part in str(value).split(","):
                        normalized = normalize_content_label(part)
                        if normalized:
                            scene_tags.add(normalized)
        categories[int(display_group_id)] = scene_tags
    return categories


def compute_groups(
    df: pd.DataFrame,
    media_mode: str,
    merge_scene_mode: bool,
    filter_people: bool,
    filter_landscapes: bool,
    filter_documents: bool,
    show_single_videos: bool,
    filter_live_photo_videos: bool,
    filter_regular_videos: bool,
) -> tuple[list[int], str]:
    display_group_col = "scene_group_id" if (media_mode == "photo" and merge_scene_mode) else "group_id"
    df["_display_group_id"] = df[display_group_col].fillna(df["group_id"]).astype(int)

    if media_mode == "photo":
        enabled_categories: set[str] = set()
        if filter_people:
            enabled_categories.add("people")
        if filter_landscapes:
            enabled_categories.add("landscapes")
        if filter_documents:
            enabled_categories.add("documents")
        categories = scene_categories(df, display_group_col)
        groups = sorted(
            group_id
            for group_id in df["_display_group_id"].unique()
            if categories.get(int(group_id), set()) & enabled_categories
        )
    else:
        groups = []
        for group_id, scene_rows in df.groupby("_display_group_id"):
            if not show_single_videos and len(scene_rows) <= 1:
                continue
            live_mask = (
                scene_rows.get("has_live_photo_pair", pd.Series(False, index=scene_rows.index))
                .fillna(False)
                .astype(bool)
            )
            has_live = bool(live_mask.any())
            has_regular = bool((~live_mask).any())
            if (has_live and filter_live_photo_videos) or (has_regular and filter_regular_videos):
                groups.append(int(group_id))
        groups = sorted(groups)
    return groups, display_group_col


def format_scene_option(df: pd.DataFrame, media_mode: str, merge_scene_mode: bool, group_value: int) -> str:
    rows = df[df["_display_group_id"] == group_value]
    strict_groups = sorted(rows["group_id"].dropna().astype(int).unique())
    merged_group = int(rows["scene_group_id"].fillna(rows["group_id"]).iloc[0])
    if media_mode == "photo" and merge_scene_mode:
        strict_label = ", ".join(str(item) for item in strict_groups)
        if strict_label == str(group_value):
            strict_label = ""
        return f"{group_value} {strict_label}".strip()
    strict_value = int(group_value)
    if merged_group == strict_value:
        return str(merged_group)
    return f"{merged_group} {strict_value}"


def split_scene_option(df: pd.DataFrame, media_mode: str, merge_scene_mode: bool, group_value: int) -> tuple[str, str]:
    rows = df[df["_display_group_id"] == group_value]
    strict_groups = sorted(rows["group_id"].dropna().astype(int).unique())
    merged_group = int(rows["scene_group_id"].fillna(rows["group_id"]).iloc[0])
    if media_mode == "photo" and merge_scene_mode:
        strict_label = ", ".join(str(item) for item in strict_groups)
        if strict_label == str(group_value):
            strict_label = ""
        return str(group_value), strict_label
    strict_value = int(group_value)
    if merged_group == strict_value:
        return str(merged_group), ""
    return str(merged_group), str(strict_value)


def format_metric_value(metric: str, value: object, row: dict[str, Any]) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if metric == "size_wh":
        w = row.get("width") or row.get("width_final")
        h = row.get("height") or row.get("height_final")
        if w is None or h is None:
            return None
        return f"{int(float(w))} × {int(float(h))} px"
    if metric in {"content_type_file", "content_type_group", "content_type_scene"}:
        return tags_to_display(value)
    if metric in {"audio_present_final", "has_live_photo_pair"}:
        return "Да" if bool(value) else "Нет"
    if metric == "duration_sec_final":
        return f"{float(value):.2f} s"
    if metric == "fps_final":
        return f"{float(value):.2f} fps"
    if metric == "bitrate_kbps_final":
        return f"{float(value):.1f} kbps"
    if metric == "live_photo_penalty":
        return f"{float(value):.2f}"
    if metric == "file_size":
        b = float(value)
        if b < 1024:
            return f"{int(b)} B"
        if b < 1024**2:
            return f"{b/1024:.1f} KB"
        return f"{b/1024**2:.2f} MB"
    if metric == "pixels":
        return f"{int(value):,} px"
    return value


def prepare_group_rows(df: pd.DataFrame, media_mode: str, group_id: int) -> pd.DataFrame:
    rows = df[df["_display_group_id"] == group_id].copy()
    if rows.empty:
        raise HTTPException(status_code=404, detail="Group not found.")
    rows["file_name"] = rows["file_path"].astype(str).map(lambda p: Path(p).name.lower())
    rows["file_stem"] = rows["file_path"].astype(str).map(lambda p: Path(p).stem.lower())
    if "is_best" in rows.columns:
        rows["best_sort"] = rows["is_best"].apply(lambda x: 0 if x else 1)
        rows = rows.sort_values(["file_stem", "file_name", "best_sort", "file_path"])
        rows = rows.drop(columns=["best_sort"])
    else:
        rows = rows.sort_values(["file_stem", "file_name", "file_path"])
    return rows


@app.get("/api/meta")
def api_meta() -> dict[str, Any]:
    return {
        "mediaModes": ["photo", "video"],
        "metricLabels": METRIC_LABELS,
        "metricHelp": METRIC_HELP,
    }


@app.get("/api/groups")
def api_groups(
    media_mode: str = Query("photo", pattern="^(photo|video)$"),
    merge_scene_mode: bool = True,
    filter_people: bool = True,
    filter_landscapes: bool = True,
    filter_documents: bool = True,
    show_single_videos: bool = False,
    filter_live_photo_videos: bool = True,
    filter_regular_videos: bool = True,
) -> dict[str, Any]:
    df = load_df(media_mode)
    groups, display_group_col = compute_groups(
        df,
        media_mode,
        merge_scene_mode,
        filter_people,
        filter_landscapes,
        filter_documents,
        show_single_videos,
        filter_live_photo_videos,
        filter_regular_videos,
    )
    if not groups:
        return {"groups": [], "displayGroupLabel": "Сцена" if media_mode == "photo" else "Видео"}
    df["_display_group_id"] = df[display_group_col].fillna(df["group_id"]).astype(int)
    payload = []
    for group_id in groups:
        rows = df[df["_display_group_id"] == group_id]
        common_label, private_label = split_scene_option(df, media_mode, merge_scene_mode, int(group_id))
        payload.append(
            {
                "id": int(group_id),
                "label": format_scene_option(df, media_mode, merge_scene_mode, int(group_id)),
                "commonLabel": common_label,
                "privateLabel": private_label,
                "size": int(len(rows)),
                "hasLivePhoto": bool(rows.get("has_live_photo_pair", pd.Series(False, index=rows.index)).fillna(False).astype(bool).any()) if media_mode == "video" else False,
            }
        )
    return {
        "groups": payload,
        "displayGroupLabel": "Сцена" if media_mode == "photo" else "Видео",
    }


@app.get("/api/group/{group_id}")
def api_group(
    group_id: int,
    media_mode: str = Query("photo", pattern="^(photo|video)$"),
    merge_scene_mode: bool = True,
    show_all_metrics: bool = False,
    filter_people: bool = True,
    filter_landscapes: bool = True,
    filter_documents: bool = True,
    show_single_videos: bool = False,
    filter_live_photo_videos: bool = True,
    filter_regular_videos: bool = True,
) -> dict[str, Any]:
    df = load_df(media_mode)
    groups, display_group_col = compute_groups(
        df,
        media_mode,
        merge_scene_mode,
        filter_people,
        filter_landscapes,
        filter_documents,
        show_single_videos,
        filter_live_photo_videos,
        filter_regular_videos,
    )
    if group_id not in groups:
        raise HTTPException(status_code=404, detail="Group not available under current filters.")
    df["_display_group_id"] = df[display_group_col].fillna(df["group_id"]).astype(int)
    rows = prepare_group_rows(df, media_mode, group_id)

    metric_cols = sorted(
        c for c in rows.columns
        if c not in {"group_id", "file_path", "is_best"} and c not in EXCLUDED_METRICS
    )
    if "width" in rows.columns and "height" in rows.columns:
        metric_cols = [c for c in metric_cols if c not in {"width", "height"}] + ["size_wh"]
    visible_metric_cols = metric_cols if show_all_metrics else [
        c for c in metric_cols
        if c == "size_wh" or len({str(v) for v in rows[c].fillna("__NA__").tolist()}) > 1
    ]

    out_rows = []
    for _, row in rows.iterrows():
        row_dict = row.to_dict()
        metrics = {}
        for metric in visible_metric_cols:
            raw_value = row_dict.get(metric)
            metrics[metric] = {
                "label": METRIC_LABELS.get(metric, metric),
                "help": METRIC_HELP.get(metric, ""),
                "value": format_metric_value(metric, raw_value, row_dict),
                "raw": None if pd.isna(raw_value) else raw_value,
            }
        out_rows.append(
            {
                "assetId": str(row_dict.get("asset_id", "")),
                "filePath": str(row_dict.get("file_path", "")),
                "fileName": Path(str(row_dict.get("file_path", ""))).name,
                "isBest": bool(row_dict.get("is_best", False)),
                "mediaType": "video" if media_mode == "video" else "photo",
                "metrics": metrics,
            }
        )

    title_suffix = ""
    if media_mode == "photo" and merge_scene_mode and "group_id" in rows.columns:
        strict_count = int(rows["group_id"].nunique())
        title_suffix = f" ({strict_count} группы)" if strict_count > 1 else ""

    return {
        "groupId": group_id,
        "title": f"{'Сцена' if media_mode == 'photo' else 'Видео'} {group_id}{title_suffix}",
        "metricOrder": visible_metric_cols,
        "rows": out_rows,
    }


@app.get("/api/media")
def api_media(path: str) -> FileResponse:
    target = Path(path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Media file not found.")
    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return FileResponse(target, media_type=media_type, filename=target.name)


@app.post("/api/delete")
def api_delete(payload: DeleteRequest) -> dict[str, Any]:
    data_path = resolve_data_path(payload.media_mode)
    df = load_df(payload.media_mode)
    if "asset_id" not in df.columns:
        raise HTTPException(status_code=500, detail="Review layer requires asset_id.")

    delete_ids = {str(item) for item in payload.asset_ids if str(item).strip()}
    if not delete_ids:
        return {"deletedAssets": 0, "deletedFiles": 0, "deletedSidecars": 0}

    selected_rows = df[df["asset_id"].astype(str).isin(delete_ids)].copy()
    deleted_files = 0
    deleted_sidecars = 0
    for _, row in selected_rows.iterrows():
        file_path = row.get("file_path")
        if pd.notna(file_path) and Path(str(file_path)).exists():
            send2trash(str(file_path))
            deleted_files += 1
        sidecar_paths = parse_sidecar_paths(row.get("sidecar_paths"))
        json_path = row.get("json_path")
        if pd.notna(json_path) and str(json_path).strip():
            sidecar_paths.append(str(json_path))
        for sidecar_path in dict.fromkeys(sidecar_paths):
            sidecar = Path(sidecar_path)
            if sidecar.exists():
                send2trash(str(sidecar))
                deleted_sidecars += 1

    df = df[~df["asset_id"].astype(str).isin(delete_ids)].copy()
    df.to_csv(data_path, index=False, encoding="utf-8-sig")
    return {
        "deletedAssets": int(len(delete_ids)),
        "deletedFiles": deleted_files,
        "deletedSidecars": deleted_sidecars,
    }


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
