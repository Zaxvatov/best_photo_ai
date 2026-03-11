import pandas as pd
import shutil
from pathlib import Path

IN_BEST = r"D:\photo_ai\data\index\best_combined_v7.csv"
IN_UNIQUE = r"D:\photo_ai\data\index\unique_media.csv"
IN_GROUPS = r"D:\photo_ai\data\index\similar_groups.csv"

OUT_DIR = Path(r"D:\photo_ai\photo_library_v7")


def safe_copy(src: Path, dest_dir: Path) -> Path:
    target = dest_dir / src.name
    i = 1
    while target.exists():
        target = dest_dir / f"{src.stem}_{i}{src.suffix}"
        i += 1
    shutil.copy2(src, target)
    return target


def copy_json_if_exists(json_path_value, dest_dir: Path):
    if not json_path_value or str(json_path_value) == "0":
        return None

    src = Path(str(json_path_value))
    if not src.exists():
        return None

    return safe_copy(src, dest_dir)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    best = pd.read_csv(IN_BEST)
    unique = pd.read_csv(IN_UNIQUE)
    groups = pd.read_csv(IN_GROUPS)

    grouped_files = set(groups["file_path"].astype(str))

    copied_media = 0
    copied_json = 0

    # 1. Лучшие из групп
    for row in best.itertuples(index=False):
        src = Path(row.file_path)
        if src.exists():
            safe_copy(src, OUT_DIR)
            copied_media += 1

        json_path = getattr(row, "json_path", None)
        if json_path:
            if copy_json_if_exists(json_path, OUT_DIR):
                copied_json += 1

    # 2. Уникальные одиночные фото, не входящие в группы
    singles = unique[~unique["file_path"].astype(str).isin(grouped_files)].copy()

    for row in singles.itertuples(index=False):
        src = Path(row.file_path)
        if src.exists():
            safe_copy(src, OUT_DIR)
            copied_media += 1

        json_path = getattr(row, "json_path", None)
        if json_path:
            if copy_json_if_exists(json_path, OUT_DIR):
                copied_json += 1

    print("media_copied =", copied_media)
    print("json_copied =", copied_json)
    print("saved_to =", OUT_DIR)


if __name__ == "__main__":
    main()