from pathlib import Path

# -----------------------
# Base directories
# -----------------------

RAW_TAKEOUT_DIR = Path(r"D:\photo_ai\data\raw_takeout")
DATA_DIR = Path(r"D:\photo_ai\data")

INDEX_DIR = DATA_DIR / "index"
LOGS_DIR = DATA_DIR / "logs"
OUTPUT_DIR = DATA_DIR / "output"

BEST_DIR = OUTPUT_DIR / "best"
REVIEW_DIR = OUTPUT_DIR / "review"


# -----------------------
# Index files
# -----------------------

MEDIA_INDEX_CSV = INDEX_DIR / "media_index.csv"

UNIQUE_MEDIA = INDEX_DIR / "unique_media.csv"
EXACT_DUPLICATES = INDEX_DIR / "exact_duplicates.csv"

ANALYSIS_IMAGES = INDEX_DIR / "analysis_images.csv"

SIMILAR_PAIRS = INDEX_DIR / "similar_pairs.csv"
SIMILAR_GROUPS = INDEX_DIR / "similar_groups.csv"


# -----------------------
# Scores
# -----------------------

SHARPNESS = INDEX_DIR / "sharpness_scores.csv"
COMPOSITION = INDEX_DIR / "composition_scores.csv"
SUBJECT = INDEX_DIR / "subject_scores.csv"
AESTHETIC = INDEX_DIR / "aesthetic_scores.csv"


# -----------------------
# Final selections
# -----------------------

BEST_COMBINED = INDEX_DIR / "best_combined.csv"
REVIEW_GROUPS = INDEX_DIR / "review_groups.csv"
