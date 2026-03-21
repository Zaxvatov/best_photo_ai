from __future__ import annotations

import json
from typing import Iterable

import pandas as pd

import config_paths as cfg


AVAILABLE_VARS = sorted(name for name in dir(cfg) if name.isupper())


def get_cfg_path(*names: str):
    for name in names:
        if hasattr(cfg, name):
            return getattr(cfg, name)
    raise ImportError(
        f"config_paths.py не содержит ни одной из переменных: {list(names)}. "
        f"Доступные переменные: {AVAILABLE_VARS}"
    )


INPUT = get_cfg_path("MEDIA_INDEX")
OUT_UNIQUE = get_cfg_path("UNIQUE_MEDIA")
OUT_DUPLICATES = get_cfg_path("EXACT_DUPLICATES")


def parse_sidecar_paths(value: object) -> list[str]:
    if pd.isna(value) or value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    return [str(parsed)]


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def truthy(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def non_empty(value: object) -> bool:
    if pd.isna(value) or value is None:
        return False
    return bool(str(value).strip())


def choose_content_type(series: pd.Series) -> str:
    values = [str(v).strip() for v in series if non_empty(v)]
    if not values:
        return ""
    counts = pd.Series(values).value_counts()
    return str(counts.index[0])


def choose_canonical_index(group: pd.DataFrame) -> int:
    work = group.copy()
    work["_has_sidecar"] = work["has_sidecar"].map(truthy) if "has_sidecar" in work else False
    work["_sidecar_count"] = pd.to_numeric(work.get("sidecar_count", 0), errors="coerce").fillna(0)
    work["_has_json_datetime"] = work.get("json_datetime", pd.Series(index=work.index, dtype=object)).map(non_empty)
    work["_has_exif_datetime"] = work.get("exif_datetime", pd.Series(index=work.index, dtype=object)).map(non_empty)
    work["_file_size"] = pd.to_numeric(work.get("file_size", 0), errors="coerce").fillna(0)
    work = work.sort_values(
        by=["_has_sidecar", "_sidecar_count", "_has_json_datetime", "_has_exif_datetime", "_file_size", "file_path"],
        ascending=[False, False, False, False, False, True],
    )
    return int(work.index[0])


def first_non_empty(group: pd.DataFrame, column: str, fallback: object = "") -> object:
    if column not in group.columns:
        return fallback
    for value in group[column]:
        if non_empty(value):
            return value
    return fallback


def canonicalize_group(group: pd.DataFrame) -> tuple[pd.Series, list[dict[str, object]]]:
    canonical_index = choose_canonical_index(group)
    canonical = group.loc[canonical_index].copy()

    merged_sidecars = unique_preserve_order(
        sidecar
        for value in group.get("sidecar_paths", pd.Series(index=group.index, dtype=object))
        for sidecar in parse_sidecar_paths(value)
    )

    canonical["sidecar_paths"] = json.dumps(merged_sidecars, ensure_ascii=False)
    canonical["sidecar_count"] = len(merged_sidecars)
    canonical["has_sidecar"] = bool(merged_sidecars)
    canonical["json_path"] = first_non_empty(group, "json_path", merged_sidecars[0] if merged_sidecars else "")
    canonical["json_datetime"] = first_non_empty(group, "json_datetime", canonical.get("json_datetime", ""))
    canonical["exif_datetime"] = first_non_empty(group, "exif_datetime", canonical.get("exif_datetime", ""))
    canonical["content_type_file"] = choose_content_type(group.get("content_type_file", pd.Series(dtype=object)))

    duplicate_rows: list[dict[str, object]] = []
    duplicate_group = group.drop(index=canonical_index)
    for row in duplicate_group.itertuples(index=False):
        duplicate_rows.append(
            {
                "canonical_asset_id": canonical.get("asset_id", ""),
                "absorbed_asset_id": getattr(row, "asset_id", ""),
                "sha256": canonical["sha256"],
                "canonical_file_path": canonical["file_path"],
                "absorbed_file_path": getattr(row, "file_path", ""),
                "canonical_sidecar_count": canonical["sidecar_count"],
                "absorbed_sidecar_count": getattr(row, "sidecar_count", 0),
                "canonical_has_sidecar": canonical["has_sidecar"],
                "absorbed_has_sidecar": getattr(row, "has_sidecar", False),
                "dedupe_action": "absorbed_into_canonical",
            }
        )

    return canonical, duplicate_rows


def ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [column for column in ["sha256", "file_path"] if column not in df.columns]
    if missing:
        raise KeyError(f"MEDIA_INDEX is missing required columns: {', '.join(missing)}")


def main():
    df = pd.read_csv(INPUT)
    ensure_required_columns(df)

    canonical_rows: list[pd.Series] = []
    duplicate_rows: list[dict[str, object]] = []

    grouped = df.groupby("sha256", dropna=False, sort=False)
    for _, group in grouped:
        canonical, duplicates = canonicalize_group(group)
        canonical_rows.append(canonical)
        duplicate_rows.extend(duplicates)

    unique = pd.DataFrame(canonical_rows).reset_index(drop=True)
    duplicates = pd.DataFrame(duplicate_rows)

    if len(duplicates) == 0:
        duplicates = pd.DataFrame(
            columns=[
                "canonical_asset_id",
                "absorbed_asset_id",
                "sha256",
                "canonical_file_path",
                "absorbed_file_path",
                "canonical_sidecar_count",
                "absorbed_sidecar_count",
                "canonical_has_sidecar",
                "absorbed_has_sidecar",
                "dedupe_action",
            ]
        )

    unique.to_csv(OUT_UNIQUE, index=False, encoding="utf-8-sig")
    duplicates.to_csv(OUT_DUPLICATES, index=False, encoding="utf-8-sig")

    duplicate_groups = int((grouped.size() > 1).sum())
    print("rows =", len(df))
    print("duplicate_rows =", len(duplicates))
    print("duplicate_groups =", duplicate_groups)
    print("unique_assets =", len(unique))
    print("saved_unique_to =", OUT_UNIQUE)
    print("saved_duplicates_to =", OUT_DUPLICATES)


if __name__ == "__main__":
    main()
