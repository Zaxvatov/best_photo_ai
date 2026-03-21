from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from config_paths import (
        BEST_COMBINED,
        BEST_DIR,
        CURATED_LIBRARY_DIR,
        CURATION_PLAN,
        MOVE_MANIFEST,
        OUTPUT_DIR,
        REVIEW_DIR,
    )
except ImportError as e:
    raise ImportError(
        "config_paths.py должен содержать переменные BEST_COMBINED, CURATION_PLAN, MOVE_MANIFEST, "
        "CURATED_LIBRARY_DIR, OUTPUT_DIR, BEST_DIR и REVIEW_DIR"
    ) from e


IN_BEST = Path(BEST_COMBINED)
PLAN_PATH = Path(CURATION_PLAN)
MANIFEST_PATH = Path(MOVE_MANIFEST)
OUT_DIR = Path(OUTPUT_DIR)
BEST_OUT_DIR = Path(BEST_DIR)
REVIEW_OUT_DIR = Path(REVIEW_DIR)
CURATED_DIR = Path(CURATED_LIBRARY_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build asset-level curation plan and optionally move best assets.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Выполнить фактическое перемещение asset в curated library. По умолчанию только строится план.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать, что будет сделано, без фактического перемещения.",
    )
    parser.add_argument(
        "--preserve-albums",
        action="store_true",
        default=True,
        help="Сохранять структуру album_path в curated library.",
    )
    return parser.parse_args()


def ensure_inputs_exist() -> None:
    if not IN_BEST.exists():
        raise FileNotFoundError(f"Отсутствует входной файл: {IN_BEST}")


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


def normalize_album_path(value: object) -> Path:
    if pd.isna(value) or value in (None, "", "0"):
        return Path()
    return Path(str(value))


def build_target_path(curated_root: Path, album_path: object, source_path: object) -> Path:
    source = Path(str(source_path))
    return curated_root / normalize_album_path(album_path) / source.name


def build_plan(df: pd.DataFrame, curated_root: Path) -> pd.DataFrame:
    work = df.copy()
    required = ["asset_id", "primary_file_path"]
    missing = [column for column in required if column not in work.columns]
    if missing:
        raise KeyError(f"best_combined.csv must contain asset-native columns: {', '.join(missing)}")
    if "sidecar_paths" not in work.columns:
        work["sidecar_paths"] = "[]"
    if "album_path" not in work.columns:
        work["album_path"] = ""

    rows: list[dict[str, object]] = []
    for row in work.itertuples(index=False):
        sidecars = parse_sidecar_paths(getattr(row, "sidecar_paths", "[]"))
        primary_path = str(getattr(row, "primary_file_path"))
        target_primary = build_target_path(curated_root, getattr(row, "album_path", ""), primary_path)
        target_sidecars = [
            str(build_target_path(curated_root, getattr(row, "album_path", ""), sidecar_path))
            for sidecar_path in sidecars
        ]
        rows.append(
            {
                "asset_id": str(getattr(row, "asset_id")),
                "group_id": getattr(row, "group_id", ""),
                "scene_group_id": getattr(row, "scene_group_id", ""),
                "selection_reason": "best_in_group",
                "primary_source_path": primary_path,
                "primary_target_path": str(target_primary),
                "sidecar_source_paths": json.dumps(sidecars, ensure_ascii=False),
                "sidecar_target_paths": json.dumps(target_sidecars, ensure_ascii=False),
                "sidecar_count": len(sidecars),
                "album_path": str(getattr(row, "album_path", "")),
                "best_score": getattr(row, "final_score", ""),
                "content_type_file": getattr(row, "content_type_file", ""),
                "content_type_group": getattr(row, "content_type_group", ""),
                "content_type_scene": getattr(row, "content_type_scene", ""),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["asset_id"])


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_move(src: Path, dst: Path) -> str:
    if not src.exists():
        return "missing_source"
    ensure_parent(dst)
    if dst.exists():
        return "target_exists"
    shutil.move(str(src), str(dst))
    return "moved"


def execute_plan(plan: pd.DataFrame) -> pd.DataFrame:
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_rows: list[dict[str, object]] = []

    for row in plan.itertuples(index=False):
        asset_id = str(row.asset_id)
        primary_src = Path(str(row.primary_source_path))
        primary_dst = Path(str(row.primary_target_path))
        primary_status = safe_move(primary_src, primary_dst)
        manifest_rows.append(
            {
                "batch_id": batch_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "asset_id": asset_id,
                "entry_kind": "primary",
                "source_path": str(primary_src),
                "target_path": str(primary_dst),
                "status": primary_status,
            }
        )

        sidecar_srcs = parse_sidecar_paths(row.sidecar_source_paths)
        sidecar_dsts = parse_sidecar_paths(row.sidecar_target_paths)
        for sidecar_src, sidecar_dst in zip(sidecar_srcs, sidecar_dsts):
            status = safe_move(Path(sidecar_src), Path(sidecar_dst))
            manifest_rows.append(
                {
                    "batch_id": batch_id,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "asset_id": asset_id,
                    "entry_kind": "sidecar",
                    "source_path": str(sidecar_src),
                    "target_path": str(sidecar_dst),
                    "status": status,
                }
            )

    return pd.DataFrame(manifest_rows)


def main() -> None:
    args = parse_args()
    ensure_inputs_exist()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    CURATED_DIR.mkdir(parents=True, exist_ok=True)

    best = pd.read_csv(IN_BEST)
    plan = build_plan(best, CURATED_DIR)
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(PLAN_PATH, index=False, encoding="utf-8-sig")

    print(f"best_source = {IN_BEST}")
    print(f"curation_plan = {PLAN_PATH}")
    print(f"curated_dir = {CURATED_DIR}")
    print(f"planned_assets = {len(plan)}")
    print(f"planned_sidecars = {int(plan['sidecar_count'].sum()) if len(plan) else 0}")

    if args.dry_run or not args.execute:
        print("execution_mode = plan_only")
        return

    manifest = execute_plan(plan)
    if MANIFEST_PATH.exists():
        existing = pd.read_csv(MANIFEST_PATH)
        manifest = pd.concat([existing, manifest], ignore_index=True)
    manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")

    moved_primary = int((manifest["entry_kind"].eq("primary") & manifest["status"].eq("moved")).sum())
    moved_sidecars = int((manifest["entry_kind"].eq("sidecar") & manifest["status"].eq("moved")).sum())
    print("execution_mode = move")
    print(f"move_manifest = {MANIFEST_PATH}")
    print(f"moved_primary = {moved_primary}")
    print(f"moved_sidecars = {moved_sidecars}")


if __name__ == "__main__":
    main()
