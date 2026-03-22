from pathlib import Path

# -----------------------
# Base directories
# -----------------------

RAW_TAKEOUT_DIR = Path(r"D:\photo_ai\data\raw_takeout")
DATA_DIR = Path(r"D:\photo_ai\data")

INDEX_DIR = DATA_DIR / "index"
LOGS_DIR = DATA_DIR / "logs"
OUTPUT_DIR = DATA_DIR / "output"
STAGING_DIR = DATA_DIR / "staging"

BEST_DIR = OUTPUT_DIR / "best"
REVIEW_DIR = OUTPUT_DIR / "review"
CURATED_LIBRARY_DIR = DATA_DIR / "library_curated"

# -----------------------
# Models
# -----------------------

# NOTE: models are stored outside DATA_DIR
# models stored inside project
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# model filenames (centralized)
AESTHETIC_MODEL_FILENAME = "sa_0_4_vit_l_14_linear.pth"

# full paths
AESTHETIC_MODEL = MODELS_DIR / AESTHETIC_MODEL_FILENAME


# -----------------------
# Index files
# -----------------------

MEDIA_ASSETS = INDEX_DIR / "media_assets.csv"
RAW_FILES_INDEX = INDEX_DIR / "raw_files.csv"
ORPHAN_SIDECARS = INDEX_DIR / "orphan_sidecars.csv"
ARCHIVES_FOUND = INDEX_DIR / "archives_found.csv"
ARCHIVES_FOUND_TXT = INDEX_DIR / "archives_found.txt"
AUDIT_REPORT = INDEX_DIR / "audit_report.csv"

UNIQUE_MEDIA = INDEX_DIR / "unique_media.csv"

EXACT_DUPLICATES = INDEX_DIR / "exact_duplicates.csv"

PHOTO_INDEX = INDEX_DIR / "photo_index.csv"
PHOTO_FEATURES = INDEX_DIR / "photo_features.csv"
PHOTO_SEMANTIC_SCORES = INDEX_DIR / "photo_semantic_scores.csv"
VIDEO_INDEX = INDEX_DIR / "video_index.csv"
VIDEO_METRICS = INDEX_DIR / "video_metrics.csv"
VIDEO_GROUPS = INDEX_DIR / "video_groups.csv"
VIDEO_BEST = INDEX_DIR / "video_best.csv"
VIDEO_REVIEW_GROUPS = INDEX_DIR / "video_review_groups.csv"
LIVE_PHOTO_CANDIDATES = INDEX_DIR / "live_photo_candidates.csv"

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
CURATION_PLAN = INDEX_DIR / "curation_plan.csv"
MOVE_MANIFEST = INDEX_DIR / "move_manifest.csv"
