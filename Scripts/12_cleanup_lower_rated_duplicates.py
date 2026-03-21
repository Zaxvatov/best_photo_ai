from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import json

try:
    from send2trash import send2trash
except ImportError:  # pragma: no cover
    send2trash = None


"""
Что делает скрипт
-----------------
Ищет точные или очень вероятные дубликаты одного и того же файла в CSV-индексе и переносит в корзину худшие копии.

Главный сценарий:
- для exact_duplicates.csv удаление возможно только когда сравнение безопасно по качеству;
- если файлы точные дубли по sha256, качество считается одинаковым, и тогда json может использоваться как tie-breaker;
- если это не точный дубль по sha256, то удаление возможно только при наличии рейтинга и только когда копия с json лучше копии без json.

Важное правило безопасности:
- group_id сам по себе НЕ считается признаком одинакового файла;
- group_id используется только как внешняя граница поиска;
- удаление возможно только внутри подгруппы с одинаковым duplicate_key.

Основной критерий совпадения — sha256.
Если sha256 нет, используется строгий составной ключ из уже посчитанных индексов: phash + file_size + width + height + extension.
Только если и этих полей нет, используется самый слабый запасной вариант: file_name целиком вместе с расширением + file_size.
JPG и HEIC с одним stem считаются разными файлами и между собой не сравниваются.

Логика удаления:
1. Сначала формируется duplicate_key:
   - сначала sha256, если колонка есть и значение заполнено
   - иначе по строгому составному ключу: phash + file_size + width + height + file_ext
   - иначе по слабому fallback: file_name целиком, включая расширение, и file_size
2. Затем формируется внешняя группа сравнения:
   - exact: только duplicate_key
   - group_id: group_id используется только как внешняя граница, но удаление всё равно идёт только внутри одинакового duplicate_key
   - auto: если есть group_id, он используется как внешняя граница, но не заменяет duplicate_key
3. Внутри безопасной подгруппы действует правило сравнения:
   - если есть копия с json и копия без json, сначала оценивается качество;
   - если это точный дубль по sha256, качество считается одинаковым;
   - если это не точный дубль по sha256, требуется рейтинг, и удаляется только копия без json с более низким рейтингом;
   - если рейтинг равен, удаление возможно только с флагом --delete-equal-rating.
4. Такие файлы переносятся в корзину.
5. Если рядом с удаляемой фотографией существует её json-sidecar, он тоже переносится в корзину.

Зачем так:
- наличие json само по себе НЕ является достаточным основанием для удаления другой копии;
- сначала нужно убедиться, что копия с json не хуже;
- для точных дублей по sha256 это условие выполняется автоматически, потому что файл побайтно тот же самый.

Ожидаемые колонки в CSV:
- file_path           (обязательно)
- json_path           (желательно)
- file_size           (обязательно)
- sha256              (желательно)
- phash               (желательно)
- width / height      (желательно)
- rating / total_score / best_score / score / final_score / file_size_n (необязательно)

По умолчанию скрипт сам пытается найти колонку рейтинга.
Если рейтинг не найден, удаление допускается только для точных дублей по sha256.
"""


DEFAULT_RATING_CANDIDATES = [
    "file_size_n",
    "rating",
    "total_score",
    "best_score",
    "score",
    "final_score",
    "file_size_n",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Удаление в корзину худших дубликатов без json-sidecar"
    )
    parser.add_argument(
        "csv_path",
        help="Путь к CSV-индексу, например review_groups.csv",
    )
    parser.add_argument(
        "--rating-column",
        default=None,
        help="Имя колонки рейтинга. Если не указано, будет автоопределение.",
    )
    parser.add_argument(
        "--group-mode",
        default="auto",
        choices=["auto", "group_id", "exact"],
        help="Как формировать группы сравнения: auto, group_id или exact.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать, что будет удалено, без фактического переноса в корзину.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Разрешить реальное удаление. Без этого флага скрипт всегда работает как dry-run.",
    )
    parser.add_argument(
        "--delete-equal-rating",
        action="store_true",
        help="Удалять даже если рейтинг равен, при наличии более предпочтительной копии с json.",
    )
    parser.add_argument(
        "--include-videos",
        action="store_true",
        help="По умолчанию видео исключаются. Этот флаг разрешает обрабатывать видео тоже.",
    )
    parser.add_argument(
        "--log-csv",
        default=None,
        help="Куда сохранить CSV-лог кандидатов на удаление.",
    )
    return parser.parse_args()



def detect_rating_column(df: pd.DataFrame, explicit_name: str | None) -> str | None:
    if explicit_name:
        if explicit_name not in df.columns:
            raise ValueError(f"Колонка рейтинга не найдена: {explicit_name}")
        return explicit_name

    for col in DEFAULT_RATING_CANDIDATES:
        if col in df.columns:
            return col

    return None



def normalize_json_exists(value: object) -> bool:
    if pd.isna(value):
        return False

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0

    text = str(value).strip()
    if not text:
        return False

    if text.lower() in {"nan", "none", "null", "false", "0"}:
        return False

    return True



def find_existing_sidecars(photo_path: Path) -> list[Path]:
    candidates = [
        photo_path.with_suffix(photo_path.suffix + ".json"),
        photo_path.parent / (photo_path.name + ".json"),
        photo_path.with_suffix(".json"),
    ]

    existing: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            existing.append(path)
    return existing


def parse_sidecar_paths(value: object) -> list[Path]:
    if pd.isna(value) or value in (None, "", 0, "0"):
        return []
    if isinstance(value, list):
        return [Path(str(item)) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return [Path(text)]
    if isinstance(parsed, list):
        return [Path(str(item)) for item in parsed if str(item).strip()]
    return [Path(str(parsed))]



def build_working_df(
    df: pd.DataFrame,
    rating_col: str | None,
    group_mode: str = "auto",
    include_videos: bool = False,
) -> pd.DataFrame:
    work = df.copy()

    required = ["file_path", "file_size"]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"В CSV отсутствуют обязательные колонки: {missing}")

    work["file_path"] = work["file_path"].astype(str).str.strip()
    work = work[work["file_path"] != ""].copy()

    if not include_videos:
        if "is_video" in work.columns:
            work = work[~work["is_video"].fillna(False)].copy()
        elif "mime_type" in work.columns:
            work = work[~work["mime_type"].astype(str).str.lower().eq("video")].copy()
        else:
            work = work[
                ~work["file_path"].str.lower().str.endswith(
                    (".mov", ".mp4", ".avi", ".mkv", ".3gp", ".webm")
                )
            ].copy()

    work["file_name"] = work["file_path"].map(lambda x: Path(x).name.lower())
    work["file_stem"] = work["file_path"].map(lambda x: Path(x).stem.lower())
    work["file_ext"] = work["file_path"].map(lambda x: Path(x).suffix.lower())
    if rating_col is not None:
        work["rating_value"] = pd.to_numeric(work[rating_col], errors="coerce")
    else:
        work["rating_value"] = pd.NA

    if "sha256" in work.columns:
        work["sha256_norm"] = work["sha256"].astype(str).str.strip().str.lower()
        work.loc[work["sha256_norm"].isin(["", "nan", "none"]), "sha256_norm"] = pd.NA
    else:
        work["sha256_norm"] = pd.NA

    if "phash" in work.columns:
        work["phash_norm"] = work["phash"].astype(str).str.strip().str.lower()
        work.loc[work["phash_norm"].isin(["", "nan", "none"]), "phash_norm"] = pd.NA
    else:
        work["phash_norm"] = pd.NA

    for dim_col in ["width", "height"]:
        if dim_col in work.columns:
            work[f"{dim_col}_norm"] = pd.to_numeric(work[dim_col], errors="coerce")
        else:
            work[f"{dim_col}_norm"] = pd.NA

    work["duplicate_key"] = work["sha256_norm"]

    strict_mask = (
        work["duplicate_key"].isna()
        & work["phash_norm"].notna()
        & work["file_size"].notna()
        & work["width_norm"].notna()
        & work["height_norm"].notna()
        & work["file_ext"].notna()
    )
    work.loc[strict_mask, "duplicate_key"] = (
        "strict::"
        + work.loc[strict_mask, "phash_norm"].astype(str)
        + "::"
        + work.loc[strict_mask, "file_size"].astype(str)
        + "::"
        + work.loc[strict_mask, "width_norm"].astype("Int64").astype(str)
        + "x"
        + work.loc[strict_mask, "height_norm"].astype("Int64").astype(str)
        + "::"
        + work.loc[strict_mask, "file_ext"].astype(str)
    )

    fallback_mask = work["duplicate_key"].isna()
    work.loc[fallback_mask, "duplicate_key"] = (
        "fallback::"
        + work.loc[fallback_mask, "file_name"].astype(str)
        + "::"
        + work.loc[fallback_mask, "file_size"].astype(str)
    )

    if "json_path" in work.columns:
        work["has_json"] = work["json_path"].map(normalize_json_exists)
    else:
        work["has_json"] = work["file_path"].map(
            lambda x: len(find_existing_sidecars(Path(x))) > 0
        )
        work["json_path"] = None

    if group_mode == "group_id":
        if "group_id" not in work.columns:
            raise ValueError("Режим --group-mode group_id требует колонку group_id в CSV")
        work["comparison_group"] = "group_id::" + work["group_id"].astype(str)
        work["comparison_group_source"] = "group_id"
    elif group_mode == "exact":
        work["comparison_group"] = work["duplicate_key"]
        work["comparison_group_source"] = "exact"
    else:
        if "group_id" in work.columns:
            group_id_text = work["group_id"].astype(str).str.strip().str.lower()
            has_group_id = ~group_id_text.isin(["", "nan", "none"])
            work["comparison_group"] = work["duplicate_key"]
            work.loc[has_group_id, "comparison_group"] = "group_id::" + work.loc[has_group_id, "group_id"].astype(str)
            work["comparison_group_source"] = "exact"
            work.loc[has_group_id, "comparison_group_source"] = "group_id"
        else:
            work["comparison_group"] = work["duplicate_key"]
            work["comparison_group_source"] = "exact"

    work["safe_subgroup"] = work["comparison_group"].astype(str) + "||" + work["duplicate_key"].astype(str)

    return work


def select_rows_to_delete(
    df: pd.DataFrame,
    delete_equal_rating: bool = False,
) -> pd.DataFrame:
    rows_to_delete: list[pd.Series] = []

    grouped = list(df.groupby(["safe_subgroup"], dropna=False, sort=False))
    total_groups = len(grouped)

    for i, (_, group) in enumerate(grouped, start=1):
        if i % 100 == 0 or i == total_groups:
            percent = (i / total_groups) * 100
            print(f"progress_groups = {percent:5.1f}% ({i}/{total_groups})", end="\r")

        if len(group) < 2:
            continue

        better_with_json = group[group["has_json"]].copy()
        if better_with_json.empty:
            continue

        candidates = group[~group["has_json"]].copy()
        if candidates.empty:
            continue

        exact_sha_match = (
            "sha256_norm" in group.columns
            and group["sha256_norm"].notna().all()
            and group["sha256_norm"].nunique(dropna=True) == 1
        )

        has_rating = (
            better_with_json["rating_value"].notna().any()
            and candidates["rating_value"].notna().any()
        )

        if exact_sha_match:
            for _, row in candidates.iterrows():
                rows_to_delete.append(row)
            continue

        if not has_rating:
            continue

        best_rating_with_json = better_with_json["rating_value"].max(skipna=True)
        if pd.isna(best_rating_with_json):
            continue

        for _, row in candidates.iterrows():
            candidate_rating = row["rating_value"]
            if pd.isna(candidate_rating):
                continue

            should_delete = (
                candidate_rating < best_rating_with_json
                if not delete_equal_rating
                else candidate_rating <= best_rating_with_json
            )

            if should_delete:
                rows_to_delete.append(row)

    print()

    if not rows_to_delete:
        return df.iloc[0:0].copy()

    result = pd.DataFrame(rows_to_delete).drop_duplicates(subset=["file_path"])
    return result.sort_values(["file_name", "file_size"], kind="stable")



def collect_paths_for_deletion(rows: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    seen_assets: set[str] = set()

    for _, row in rows.iterrows():
        asset_key = str(row.get("asset_id") or row.get("file_path", "")).strip()
        if asset_key and asset_key in seen_assets:
            continue
        if asset_key:
            seen_assets.add(asset_key)

        photo_path = Path(str(row["file_path"]))

        candidates = [photo_path]

        sidecar_paths = parse_sidecar_paths(row.get("sidecar_paths"))
        json_path = row.get("json_path")
        if normalize_json_exists(json_path):
            sidecar_paths.append(Path(str(json_path)))
        if not sidecar_paths:
            sidecar_paths.extend(find_existing_sidecars(photo_path))
        candidates.extend(sidecar_paths)

        for path in candidates:
            resolved = path.resolve(strict=False)
            if resolved in seen:
                continue
            seen.add(resolved)
            if path.exists():
                paths.append(path)

    return paths



def print_report(rows_to_delete: pd.DataFrame, rating_col: str | None, dry_run: bool) -> None:
    mode = "DRY RUN" if dry_run else "DELETE (EXECUTE)"
    print(f"mode = {mode}")
    print(f"rating_column = {rating_col if rating_col is not None else 'None'}")
    print(f"photos_to_delete = {len(rows_to_delete)}")
    if "is_video" in rows_to_delete.columns and not rows_to_delete.empty:
        print(f"videos_to_delete = {int(rows_to_delete['is_video'].fillna(False).sum())}")
    if not rows_to_delete.empty and "comparison_group_source" in rows_to_delete.columns:
        counts = rows_to_delete["comparison_group_source"].value_counts(dropna=False).to_dict()
        print(f"comparison_group_source_counts = {counts}")

    if rows_to_delete.empty:
        return

    show_cols = [
        c
        for c in ["comparison_group_source", "comparison_group", "safe_subgroup", "group_id", "file_name", "file_ext", "file_size", "width_norm", "height_norm", "sha256_norm", "phash_norm", "duplicate_key", "rating_value", "has_json", "file_path", "json_path"]
        if c in rows_to_delete.columns
    ]
    print(rows_to_delete[show_cols].to_string(index=False))



def move_to_trash(paths: Iterable[Path], dry_run: bool) -> tuple[int, int]:
    moved = 0
    missing = 0

    paths = list(paths)
    total = len(paths)

    for i, path in enumerate(paths, start=1):
        percent = (i / total) * 100 if total else 100
        if i % 50 == 0 or i == total:
            label = "progress_candidates" if dry_run else "progress_delete"
            print(f"{label} = {percent:5.1f}% ({i}/{total})", end="\r")

        if not path.exists():
            missing += 1
            print(f"missing = {path}")
            continue

        if dry_run:
            moved += 1
            continue

        send2trash(str(path))
        moved += 1

    print()
    return moved, missing



def default_log_path(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + "_delete_candidates.csv")



def main() -> int:
    args = parse_args()

    if not args.execute:
        args.dry_run = True

    if send2trash is None and not args.dry_run:
        print("Не установлен пакет send2trash. Установи его командой: pip install send2trash", file=sys.stderr)
        return 1

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"CSV не найден: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    rating_col = detect_rating_column(df, args.rating_column)
    work = build_working_df(
        df,
        rating_col,
        group_mode=args.group_mode,
        include_videos=args.include_videos,
    )
    rows_to_delete = select_rows_to_delete(
        work,
        delete_equal_rating=args.delete_equal_rating,
    )

    print_report(rows_to_delete, rating_col, args.dry_run)

    log_path = Path(args.log_csv) if args.log_csv else default_log_path(csv_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows_to_delete.to_csv(log_path, index=False)
    print(f"log_csv_saved = {log_path}")

    paths = collect_paths_for_deletion(rows_to_delete)
    moved, missing = move_to_trash(paths, dry_run=args.dry_run)

    print(f"paths_processed = {len(paths)}")
    if args.dry_run:
        print(f"would_move_to_trash = {moved}")
    else:
        print(f"moved_to_trash = {moved}")
    print(f"missing_paths = {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
