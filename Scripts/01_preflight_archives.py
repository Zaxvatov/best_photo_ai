from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config_paths


RAW_TAKEOUT_DIR = Path(config_paths.RAW_TAKEOUT_DIR)
ARCHIVES_FOUND = Path(config_paths.ARCHIVES_FOUND)
AUDIT_REPORT = Path(config_paths.AUDIT_REPORT)

ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"}


@dataclass
class ArchiveRecord:
    file_path: str
    file_name: str
    extension: str
    file_size: int
    created_at_fs: float
    album_path: str


def collect_archives(root: Path) -> list[ArchiveRecord]:
    rows: list[ArchiveRecord] = []
    for path in tqdm(root.rglob("*"), desc="Поиск архивов"):
        if not path.is_file() or path.suffix.lower() not in ARCHIVE_EXTENSIONS:
            continue
        stat = path.stat()
        try:
            album_path = str(path.parent.relative_to(root))
        except ValueError:
            album_path = str(path.parent)
        rows.append(
            ArchiveRecord(
                file_path=str(path),
                file_name=path.name,
                extension=path.suffix.lower(),
                file_size=stat.st_size,
                created_at_fs=stat.st_mtime,
                album_path=album_path,
            )
        )
    return rows


def merge_audit(archive_count: int) -> None:
    rows = [{"metric": "archives_found_preflight", "value": archive_count}]
    if AUDIT_REPORT.exists():
        try:
            current = pd.read_csv(AUDIT_REPORT)
            rows.extend(current.to_dict("records"))
        except Exception:
            pass
    audit_df = pd.DataFrame(rows).drop_duplicates(subset=["metric"], keep="first")
    AUDIT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(AUDIT_REPORT, index=False, encoding="utf-8-sig")


def main() -> int:
    if not RAW_TAKEOUT_DIR.exists():
        raise FileNotFoundError(f"Не найден RAW_TAKEOUT_DIR: {RAW_TAKEOUT_DIR}")
    if not RAW_TAKEOUT_DIR.is_dir():
        raise NotADirectoryError(f"RAW_TAKEOUT_DIR не является папкой: {RAW_TAKEOUT_DIR}")

    ARCHIVES_FOUND.parent.mkdir(parents=True, exist_ok=True)

    archives = collect_archives(RAW_TAKEOUT_DIR)
    archives_df = pd.DataFrame([asdict(row) for row in archives])
    if len(archives_df) == 0:
        archives_df = pd.DataFrame(columns=list(ArchiveRecord.__annotations__.keys()))
    archives_df.to_csv(ARCHIVES_FOUND, index=False, encoding="utf-8-sig")

    merge_audit(len(archives))

    print(f"takeout_dir = {RAW_TAKEOUT_DIR}")
    print(f"archives_found = {len(archives)}")
    print(f"saved_archives_to = {ARCHIVES_FOUND}")
    if archives:
        print("archive_paths:")
        for path in archives_df["file_path"].tolist():
            print(path)
    else:
        print("archive_paths: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
