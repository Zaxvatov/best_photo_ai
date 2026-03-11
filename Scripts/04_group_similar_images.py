import pandas as pd
import imagehash
from pathlib import Path
from collections import defaultdict

INPUT = r"D:\photo_ai\data\index\analysis_images.csv"
OUT_PAIRS = r"D:\photo_ai\data\index\similar_pairs.csv"
OUT_GROUPS = r"D:\photo_ai\data\index\similar_groups.csv"

# Чем меньше порог, тем строже похожесть
PHASH_DISTANCE_THRESHOLD = 5

# Бакетизация по префиксу pHash для ускорения
# 4 символа = хороший компромисс
PHASH_PREFIX_LEN = 4


def hamming_distance_hex(h1: str, h2: str) -> int:
    return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    buckets = defaultdict(list)

    for row in df.itertuples(index=False):
        phash = str(row.phash)
        prefix = phash[:PHASH_PREFIX_LEN]
        buckets[prefix].append((row.file_path, phash))

    pairs = []

    for prefix, items in buckets.items():
        n = len(items)
        if n < 2:
            continue

        for i in range(n):
            file1, phash1 = items[i]
            for j in range(i + 1, n):
                file2, phash2 = items[j]

                d = hamming_distance_hex(phash1, phash2)
                if d <= PHASH_DISTANCE_THRESHOLD:
                    pairs.append((file1, file2, d))

    return pd.DataFrame(pairs, columns=["img1", "img2", "distance"])


def build_groups(pairs: pd.DataFrame) -> pd.DataFrame:
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in zip(pairs["img1"], pairs["img2"]):
        union(a, b)

    groups = defaultdict(list)

    for x in parent:
        root = find(x)
        groups[root].append(x)

    rows = []
    gid = 0

    for files in groups.values():
        if len(files) > 1:
            gid += 1
            for f in sorted(files):
                rows.append((gid, f))

    return pd.DataFrame(rows, columns=["group_id", "file_path"])


def main():
    df = pd.read_csv(INPUT)

    # Берём только записи с phash
    df = df[df["phash"].notna()].copy()

    print("images_for_grouping =", len(df))

    pairs = build_pairs(df)
    pairs.to_csv(OUT_PAIRS, index=False, encoding="utf-8-sig")

    print("pairs_found =", len(pairs))
    print("pairs_saved_to =", OUT_PAIRS)

    groups = build_groups(pairs)
    groups.to_csv(OUT_GROUPS, index=False, encoding="utf-8-sig")

    print("groups_found =", groups["group_id"].nunique() if len(groups) else 0)
    print("files_in_groups =", len(groups))
    print("groups_saved_to =", OUT_GROUPS)


if __name__ == "__main__":
    main()