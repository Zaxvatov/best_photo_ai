from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config_paths


RAW_TAKEOUT_DIR = Path(config_paths.RAW_TAKEOUT_DIR)
ARCHIVES_FOUND = Path(config_paths.ARCHIVES_FOUND)
ARCHIVES_FOUND_TXT = Path(config_paths.ARCHIVES_FOUND_TXT)
AUDIT_REPORT = Path(config_paths.AUDIT_REPORT)
STAGING_DIR = Path(config_paths.STAGING_DIR)

ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"}
SUPPORTED_UNPACK_EXTENSIONS = {".zip", ".tar", ".gz", ".bz2", ".xz"}
STOP_EXIT_CODE = 40


@dataclass
class ArchiveRecord:
    file_path: str
    file_name: str
    extension: str
    file_size: int
    created_at_fs: float
    album_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive preflight before asset indexing.")
    parser.add_argument("takeout_dir", nargs="?", default=str(RAW_TAKEOUT_DIR))
    parser.add_argument(
        "--mode",
        choices=["interactive", "continue", "stop", "unpack"],
        default="interactive",
        help="Поведение при найденных архивах: interactive, continue, stop, unpack.",
    )
    parser.add_argument(
        "--save-list",
        action="store_true",
        help="Сохранить список найденных архивов в txt.",
    )
    return parser.parse_args()


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


def save_archive_csv(archives: list[ArchiveRecord]) -> pd.DataFrame:
    ARCHIVES_FOUND.parent.mkdir(parents=True, exist_ok=True)
    archives_df = pd.DataFrame([asdict(row) for row in archives])
    if len(archives_df) == 0:
        archives_df = pd.DataFrame(columns=list(ArchiveRecord.__annotations__.keys()))
    archives_df.to_csv(ARCHIVES_FOUND, index=False, encoding="utf-8-sig")
    return archives_df


def save_archive_txt(archives_df: pd.DataFrame) -> Path:
    ARCHIVES_FOUND_TXT.parent.mkdir(parents=True, exist_ok=True)
    lines = archives_df["file_path"].astype(str).tolist() if len(archives_df) else []
    ARCHIVES_FOUND_TXT.write_text("\n".join(lines), encoding="utf-8")
    return ARCHIVES_FOUND_TXT


def merge_audit(archive_count: int, action: str, unpacked_count: int, skipped_unpack_count: int) -> None:
    rows = [
        {"metric": "archives_found_preflight", "value": archive_count},
        {"metric": "archives_preflight_action", "value": action},
        {"metric": "archives_unpacked_preflight", "value": unpacked_count},
        {"metric": "archives_unpack_skipped_preflight", "value": skipped_unpack_count},
    ]
    if AUDIT_REPORT.exists():
        try:
            current = pd.read_csv(AUDIT_REPORT)
            rows.extend(current.to_dict("records"))
        except Exception:
            pass
    audit_df = pd.DataFrame(rows).drop_duplicates(subset=["metric"], keep="first")
    AUDIT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(AUDIT_REPORT, index=False, encoding="utf-8-sig")


def print_archive_list(archives_df: pd.DataFrame) -> None:
    print("archive_paths:")
    if archives_df.empty:
        print("none")
        return
    for path in archives_df["file_path"].tolist():
        print(path)


def prompt_action() -> str:
    print()
    print("Обнаружены архивы. Выберите действие:")
    print("  [1] stop      - остановить пайплайн")
    print("  [2] continue  - игнорировать архивы и продолжить")
    print("  [3] save      - сохранить список архивов в txt и остановить")
    print("  [4] unpack    - распаковать поддерживаемые архивы в staging и продолжить")
    while True:
        choice = input("Введите 1/2/3/4: ").strip().lower()
        mapping = {
            "1": "stop",
            "2": "continue",
            "3": "save",
            "4": "unpack",
            "stop": "stop",
            "continue": "continue",
            "save": "save",
            "unpack": "unpack",
        }
        if choice in mapping:
            return mapping[choice]
        print("Неверный выбор. Ожидалось 1, 2, 3 или 4.")


def unpack_archives(root: Path, archives_df: pd.DataFrame) -> tuple[int, int]:
    unpack_root = STAGING_DIR / "archives_unpacked"
    if unpack_root.exists():
        shutil.rmtree(unpack_root)
    unpack_root.mkdir(parents=True, exist_ok=True)

    unpacked_count = 0
    skipped_count = 0
    for row in archives_df.itertuples(index=False):
        archive_path = Path(str(row.file_path))
        extension = str(row.extension).lower()
        if extension not in SUPPORTED_UNPACK_EXTENSIONS:
            skipped_count += 1
            continue

        try:
            relative_parent = archive_path.parent.relative_to(root)
        except ValueError:
            relative_parent = Path()

        target_dir = unpack_root / relative_parent / archive_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            shutil.unpack_archive(str(archive_path), str(target_dir))
            unpacked_count += 1
        except (shutil.ReadError, ValueError):
            skipped_count += 1
    return unpacked_count, skipped_count


def main() -> int:
    args = parse_args()
    takeout_dir = Path(args.takeout_dir)

    if not takeout_dir.exists():
        raise FileNotFoundError(f"Не найден RAW_TAKEOUT_DIR: {takeout_dir}")
    if not takeout_dir.is_dir():
        raise NotADirectoryError(f"RAW_TAKEOUT_DIR не является папкой: {takeout_dir}")

    archives = collect_archives(takeout_dir)
    archives_df = save_archive_csv(archives)

    print(f"takeout_dir = {takeout_dir}")
    print(f"archives_found = {len(archives)}")
    print(f"saved_archives_to = {ARCHIVES_FOUND}")
    print_archive_list(archives_df)

    if not archives:
        merge_audit(0, "none", 0, 0)
        return 0

    if args.save_list:
        saved_txt = save_archive_txt(archives_df)
        print(f"saved_archives_txt_to = {saved_txt}")

    action = args.mode
    if action == "interactive":
        action = prompt_action()

    unpacked_count = 0
    skipped_unpack_count = 0

    if action == "save":
        saved_txt = save_archive_txt(archives_df)
        print(f"saved_archives_txt_to = {saved_txt}")
        merge_audit(len(archives), "save_and_stop", 0, 0)
        print("preflight_action = stop")
        return STOP_EXIT_CODE

    if action == "stop":
        merge_audit(len(archives), "stop", 0, 0)
        print("preflight_action = stop")
        return STOP_EXIT_CODE

    if action == "unpack":
        unpacked_count, skipped_unpack_count = unpack_archives(takeout_dir, archives_df)
        print(f"staging_unpack_dir = {STAGING_DIR / 'archives_unpacked'}")
        print(f"archives_unpacked = {unpacked_count}")
        print(f"archives_unpack_skipped = {skipped_unpack_count}")
        merge_audit(len(archives), "unpack", unpacked_count, skipped_unpack_count)
        print("preflight_action = continue")
        return 0

    merge_audit(len(archives), "continue", 0, 0)
    print("preflight_action = continue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
