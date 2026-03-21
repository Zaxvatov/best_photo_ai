from __future__ import annotations

import math
import re
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import imagehash
import numpy as np
import open_clip
import pandas as pd
import torch
import cv2
from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm

import config_paths as cfg

PHOTO_INDEX = Path(cfg.PHOTO_INDEX)
SIMILAR_PAIRS = Path(cfg.SIMILAR_PAIRS)
SIMILAR_GROUPS = Path(cfg.SIMILAR_GROUPS)

# Stage 1: near-exact grouping via pHash
PHASH_DISTANCE_THRESHOLD = 5
PHASH_PREFIX_LEN = 4

# Stage 2: same-scene rescue using CLIP in a local neighborhood
CLIP_PHASH_DISTANCE_MAX = 24
CLIP_TIME_WINDOW_SECONDS = 4 * 60 * 60
CLIP_FILENAME_GAP_MAX = 80
CLIP_NEIGHBOR_LOOKAHEAD = 40
CLIP_ASPECT_RATIO_DELTA_MAX = 0.12
SCENE_ASPECT_RATIO_DELTA_MAX = 0.75
CLIP_SIMILARITY_THRESHOLD = 0.965
SCENE_TIME_WINDOW_SECONDS = 120
SCENE_FILENAME_GAP_MAX = 25
SCENE_PHASH_DISTANCE_MAX = 40
SCENE_CLIP_THRESHOLD = 0.94
SCENE_BACKGROUND_THRESHOLD = 0.88
SCENE_BRIDGE_CLIP_THRESHOLD = 0.90
SCENE_BRIDGE_BACKGROUND_THRESHOLD = 0.93
SCENE_BRIDGE_PHASH_DISTANCE_MAX = 12
SCENE_WIDE_BRIDGE_CLIP_THRESHOLD = 0.91
SCENE_WIDE_BRIDGE_BACKGROUND_THRESHOLD = 0.90
SCENE_WIDE_BRIDGE_PHASH_DISTANCE_MAX = 28
FACE_CROP_MIN_SIZE = 80
FACE_CROP_MIN_AREA = 30_000
FACE_CROP_MAX_COUNT = 4
FACE_IDENTITY_VETO_THRESHOLD = 0.88
FACE_IDENTITY_VETO_BACKGROUND_MIN = 0.90
FACE_IDENTITY_VETO_CLIP_MAX = 0.975
FACE_IDENTITY_VETO_PHASH_MAX = 22
BURST_TIME_WINDOW_SECONDS = 5
BURST_FILENAME_GAP_MAX = 15
BURST_PHASH_DISTANCE_MAX = 18
BURST_CLIP_THRESHOLD = 0.95
BURST_BACKGROUND_THRESHOLD = 0.9
CROSS_ALBUM_TIME_WINDOW_SECONDS = 90
CROSS_ALBUM_FILENAME_GAP_MAX = 10
CROSS_ALBUM_PHASH_DISTANCE_MAX = 18
CROSS_ALBUM_CLIP_THRESHOLD = 0.90
CROSS_ALBUM_BACKGROUND_THRESHOLD = 0.86
CROSS_ALBUM_NEIGHBOR_LOOKAHEAD = 20
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# Offline-safe fallback if CLIP weights are unavailable
ORB_SCORE_THRESHOLD = 0.62
ORB_NFEATURES = 1200
ORB_MAX_DIM = 960
CLIP_BATCH_SIZE_CUDA = 64
CLIP_BATCH_SIZE_CPU = 16

EXIF_DT_FORMAT = "%Y:%m:%d %H:%M:%S"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}
FILENAME_SEQ_RE = re.compile(r"(\d+)(?!.*\d)")

register_heif_opener()


@dataclass(frozen=True)
class CandidatePair:
    img1: str
    img2: str
    phash_distance: int
    clip_threshold: float
    background_threshold: float | None


@dataclass(frozen=True)
class SimilarPair:
    img1: str
    img2: str
    distance: int
    match_source: str
    similarity: float
    pair_scope: str


def hamming_distance_hex(h1: str, h2: str) -> int:
    return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)


def safe_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_capture_ts(row: pd.Series) -> float:
    exif_dt = row.get("exif_datetime")
    if isinstance(exif_dt, str) and exif_dt.strip():
        try:
            return datetime.strptime(exif_dt.strip(), EXIF_DT_FORMAT).timestamp()
        except ValueError:
            pass

    json_ts = safe_float(row.get("json_datetime"))
    if json_ts is not None and not math.isnan(json_ts):
        return json_ts

    created_ts = safe_float(row.get("created_at_fs"))
    if created_ts is not None and not math.isnan(created_ts):
        return created_ts

    return 0.0


def parse_filename_seq(file_name: str) -> int | None:
    match = FILENAME_SEQ_RE.search(file_name)
    return int(match.group(1)) if match else None


def prepare_metadata(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["file_name"] = work["file_path"].map(lambda p: Path(str(p)).name)
    work["capture_ts"] = work.apply(parse_capture_ts, axis=1)
    work["file_seq"] = work["file_name"].map(parse_filename_seq)
    work["aspect_ratio"] = work["width"].astype(float) / work["height"].replace(0, np.nan).astype(float)
    work["album_key"] = work["album_path"].fillna("").astype(str).str.lower()
    return work


def build_phash_pairs(df: pd.DataFrame) -> list[SimilarPair]:
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for row in df.itertuples(index=False):
        phash = str(row.phash)
        buckets[phash[:PHASH_PREFIX_LEN]].append((row.file_path, phash))

    pairs: list[SimilarPair] = []

    for items in buckets.values():
        n = len(items)
        if n < 2:
            continue

        for i in range(n):
            file1, phash1 = items[i]
            for j in range(i + 1, n):
                file2, phash2 = items[j]
                distance = hamming_distance_hex(phash1, phash2)
                if distance <= PHASH_DISTANCE_THRESHOLD:
                    pairs.append(
                        SimilarPair(
                            img1=file1,
                            img2=file2,
                            distance=distance,
                            match_source="phash",
                            similarity=float(1.0 - distance / 64.0),
                            pair_scope="strict",
                        )
                    )

    return pairs


def build_local_phash_pairs(df: pd.DataFrame, existing_pairs: list[SimilarPair]) -> list[SimilarPair]:
    existing_edges = {frozenset((pair.img1, pair.img2)) for pair in existing_pairs}
    work = df.sort_values(["album_key", "capture_ts", "file_name"]).reset_index(drop=True)
    pairs: list[SimilarPair] = []

    for _, group in work.groupby("album_key", sort=False):
        rows = list(group.itertuples(index=False))
        n = len(rows)
        for i in range(n):
            left = rows[i]
            for j in range(i + 1, min(n, i + 1 + CLIP_NEIGHBOR_LOOKAHEAD)):
                right = rows[j]

                key = frozenset((str(left.file_path), str(right.file_path)))
                if key in existing_edges:
                    continue

                time_diff = abs(float(right.capture_ts) - float(left.capture_ts))
                left_seq = left.file_seq if pd.notna(left.file_seq) else None
                right_seq = right.file_seq if pd.notna(right.file_seq) else None
                seq_diff = abs(int(right_seq) - int(left_seq)) if left_seq is not None and right_seq is not None else None

                if time_diff > SCENE_TIME_WINDOW_SECONDS and (seq_diff is None or seq_diff > SCENE_FILENAME_GAP_MAX):
                    continue

                phash_distance = hamming_distance_hex(str(left.phash), str(right.phash))
                if phash_distance > PHASH_DISTANCE_THRESHOLD:
                    continue

                pairs.append(
                    SimilarPair(
                        img1=str(left.file_path),
                        img2=str(right.file_path),
                        distance=phash_distance,
                        match_source="phash_local",
                        similarity=float(1.0 - phash_distance / 64.0),
                        pair_scope="strict",
                    )
                )
                existing_edges.add(key)

    return pairs


def iter_clip_candidates(df: pd.DataFrame, phash_edges: set[frozenset[str]]) -> Iterable[CandidatePair]:
    work = df.sort_values(["album_key", "capture_ts", "file_name"]).reset_index(drop=True)
    emitted: set[frozenset[str]] = set()

    for _, group in work.groupby("album_key", sort=False):
        rows = list(group.itertuples(index=False))
        n = len(rows)
        for i in range(n):
            left = rows[i]
            for j in range(i + 1, min(n, i + 1 + CLIP_NEIGHBOR_LOOKAHEAD)):
                right = rows[j]

                if frozenset((left.file_path, right.file_path)) in phash_edges:
                    continue

                time_diff = abs(float(right.capture_ts) - float(left.capture_ts))
                left_seq = left.file_seq if pd.notna(left.file_seq) else None
                right_seq = right.file_seq if pd.notna(right.file_seq) else None
                seq_diff = abs(int(right_seq) - int(left_seq)) if left_seq is not None and right_seq is not None else None

                if time_diff > CLIP_TIME_WINDOW_SECONDS and (seq_diff is None or seq_diff > CLIP_FILENAME_GAP_MAX):
                    continue

                clip_threshold = CLIP_SIMILARITY_THRESHOLD
                phash_distance_max = CLIP_PHASH_DISTANCE_MAX
                phash_distance = hamming_distance_hex(str(left.phash), str(right.phash))
                aspect_ratio_delta_max = CLIP_ASPECT_RATIO_DELTA_MAX

                if time_diff <= SCENE_TIME_WINDOW_SECONDS and seq_diff is not None and seq_diff <= SCENE_FILENAME_GAP_MAX:
                    clip_threshold = SCENE_CLIP_THRESHOLD
                    phash_distance_max = SCENE_PHASH_DISTANCE_MAX
                    aspect_ratio_delta_max = SCENE_ASPECT_RATIO_DELTA_MAX
                    background_threshold = SCENE_BACKGROUND_THRESHOLD
                else:
                    background_threshold = None

                if time_diff <= BURST_TIME_WINDOW_SECONDS and seq_diff is not None and seq_diff <= BURST_FILENAME_GAP_MAX:
                    if phash_distance <= BURST_PHASH_DISTANCE_MAX:
                        clip_threshold = BURST_CLIP_THRESHOLD
                        phash_distance_max = max(phash_distance_max, BURST_PHASH_DISTANCE_MAX)
                        background_threshold = BURST_BACKGROUND_THRESHOLD

                if pd.notna(left.aspect_ratio) and pd.notna(right.aspect_ratio):
                    if abs(float(left.aspect_ratio) - float(right.aspect_ratio)) > aspect_ratio_delta_max:
                        continue

                if phash_distance <= PHASH_DISTANCE_THRESHOLD or phash_distance > phash_distance_max:
                    continue

                key = frozenset((str(left.file_path), str(right.file_path)))
                if key in emitted:
                    continue
                emitted.add(key)

                yield CandidatePair(
                    img1=str(left.file_path),
                    img2=str(right.file_path),
                    phash_distance=phash_distance,
                    clip_threshold=clip_threshold,
                    background_threshold=background_threshold,
                )

    global_rows = list(df.sort_values(["capture_ts", "file_name"]).itertuples(index=False))
    total = len(global_rows)
    for i in range(total):
        left = global_rows[i]
        for j in range(i + 1, min(total, i + 1 + CROSS_ALBUM_NEIGHBOR_LOOKAHEAD)):
            right = global_rows[j]

            if left.album_key == right.album_key:
                continue

            time_diff = abs(float(right.capture_ts) - float(left.capture_ts))
            if time_diff > CROSS_ALBUM_TIME_WINDOW_SECONDS:
                break

            left_seq = left.file_seq if pd.notna(left.file_seq) else None
            right_seq = right.file_seq if pd.notna(right.file_seq) else None
            if left_seq is None or right_seq is None:
                continue

            seq_diff = abs(int(right_seq) - int(left_seq))
            if seq_diff > CROSS_ALBUM_FILENAME_GAP_MAX:
                continue

            if pd.notna(left.aspect_ratio) and pd.notna(right.aspect_ratio):
                if abs(float(left.aspect_ratio) - float(right.aspect_ratio)) > CLIP_ASPECT_RATIO_DELTA_MAX:
                    continue

            phash_distance = hamming_distance_hex(str(left.phash), str(right.phash))
            if phash_distance <= PHASH_DISTANCE_THRESHOLD or phash_distance > CROSS_ALBUM_PHASH_DISTANCE_MAX:
                continue

            key = frozenset((str(left.file_path), str(right.file_path)))
            if key in phash_edges or key in emitted:
                continue
            emitted.add(key)

            yield CandidatePair(
                img1=str(left.file_path),
                img2=str(right.file_path),
                phash_distance=phash_distance,
                clip_threshold=CROSS_ALBUM_CLIP_THRESHOLD,
                background_threshold=CROSS_ALBUM_BACKGROUND_THRESHOLD,
            )


class ClipFeatureStore:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = CLIP_BATCH_SIZE_CUDA if self.device == "cuda" else CLIP_BATCH_SIZE_CPU
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED,
            device=self.device,
        )
        self.model.eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self.cache: dict[str, dict[str, torch.Tensor]] = {}
        self.face_cache: dict[str, list[tuple[float, torch.Tensor]]] = {}
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.cache_dir = PHOTO_INDEX.parent / f"clip_cache_{CLIP_MODEL_NAME}_{CLIP_PRETRAINED}".replace("-", "_").replace("/", "_")
        self.embed_cache_dir = self.cache_dir / "embed"
        self.face_cache_dir = self.cache_dir / "faces"
        self.embed_cache_dir.mkdir(parents=True, exist_ok=True)
        self.face_cache_dir.mkdir(parents=True, exist_ok=True)
        self.embed_cache_hits = 0
        self.embed_cache_misses = 0
        self.face_cache_hits = 0
        self.face_cache_misses = 0

    def _cache_key(self, file_path: str, suffix: str) -> Path:
        path = Path(file_path)
        stat = path.stat()
        signature = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{CLIP_MODEL_NAME}|{CLIP_PRETRAINED}|{suffix}"
        digest = hashlib.sha1(signature.encode("utf-8", errors="ignore")).hexdigest()
        base_dir = self.embed_cache_dir if suffix == "embed" else self.face_cache_dir
        return base_dir / f"{digest}.pt"

    def _crop_views(self, image: Image.Image) -> dict[str, Image.Image]:
        w, h = image.size
        return {
            "full": image,
            "top": image.crop((0, 0, w, max(1, int(h * 0.35)))),
            "left": image.crop((0, 0, max(1, int(w * 0.28)), h)),
            "right": image.crop((min(w - 1, int(w * 0.72)), 0, w, h)),
        }

    def _encode_batch(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        batch = torch.stack(tensors, dim=0).to(self.device, non_blocking=self.device == "cuda")
        if self.device == "cuda":
            batch = batch.half()
        with torch.inference_mode():
            encoded = self.model.encode_image(batch)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
        return encoded.detach().cpu()

    def precompute(self, file_paths: Iterable[str]) -> None:
        missing_paths: list[str] = []
        for file_path in file_paths:
            if file_path in self.cache:
                continue
            cache_path = self._cache_key(file_path, "embed")
            if cache_path.exists():
                try:
                    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                    if isinstance(payload, dict) and payload:
                        restored = {
                            str(key): value.detach().cpu() if isinstance(value, torch.Tensor) else torch.tensor(value)
                            for key, value in payload.items()
                        }
                        self.cache[file_path] = restored
                        self.embed_cache_hits += 1
                        continue
                except Exception:
                    pass
            missing_paths.append(file_path)

        print("clip_unique_images =", len(set(file_paths)))
        print("clip_precompute_misses =", len(missing_paths))
        if not missing_paths:
            return

        view_names: list[str] = []
        view_paths: list[str] = []
        tensors: list[torch.Tensor] = []
        pending: list[tuple[str, str]] = []
        progress = tqdm(total=len(missing_paths), desc="CLIP embeddings", unit="image")

        def flush() -> None:
            nonlocal view_names, view_paths, tensors, pending
            if not tensors:
                return
            encoded = self._encode_batch(tensors)
            grouped: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
            for idx, (path_key, view_key) in enumerate(pending):
                grouped[path_key][view_key] = encoded[idx]
            for path_key, features in grouped.items():
                self.cache[path_key] = features
                self.embed_cache_misses += 1
                try:
                    torch.save(features, self._cache_key(path_key, "embed"))
                except Exception:
                    pass
                progress.update(1)
            view_names = []
            view_paths = []
            tensors = []
            pending = []

        max_views_per_batch = max(4, self.batch_size * 4)
        for file_path in missing_paths:
            with Image.open(file_path) as img:
                image = img.convert("RGB")
            crops = self._crop_views(image)
            for view_key, crop in crops.items():
                tensors.append(self.preprocess(crop))
                pending.append((file_path, view_key))
            if len(tensors) >= max_views_per_batch:
                flush()
        flush()
        progress.close()

    def embed(self, file_path: str) -> dict[str, torch.Tensor]:
        cached = self.cache.get(file_path)
        if cached is not None:
            return cached

        cache_path = self._cache_key(file_path, "embed")
        if cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                if isinstance(payload, dict) and payload:
                    restored = {
                        str(key): value.detach().cpu() if isinstance(value, torch.Tensor) else torch.tensor(value)
                        for key, value in payload.items()
                    }
                    self.cache[file_path] = restored
                    self.embed_cache_hits += 1
                    return restored
            except Exception:
                pass

        with Image.open(file_path) as img:
            image = img.convert("RGB")
        crops = self._crop_views(image)
        tensors = [self.preprocess(crop) for crop in crops.values()]
        encoded = self._encode_batch(tensors)
        features = {
            key: encoded[idx]
            for idx, key in enumerate(crops.keys())
        }

        self.cache[file_path] = features
        self.embed_cache_misses += 1
        try:
            torch.save(features, cache_path)
        except Exception:
            pass
        return features

    def _background_similarity(self, feat1: dict[str, torch.Tensor], feat2: dict[str, torch.Tensor]) -> float:
        bg_scores = [
            float(torch.dot(feat1["top"], feat2["top"]).item()),
            float(torch.dot(feat1["left"], feat2["left"]).item()),
            float(torch.dot(feat1["right"], feat2["right"]).item()),
        ]
        bg_scores.sort()
        return float(sum(bg_scores[-2:]) / 2.0)

    def similarity(self, file1: str, file2: str) -> tuple[float, float]:
        feat1 = self.embed(file1)
        feat2 = self.embed(file2)
        full = float(torch.dot(feat1["full"], feat2["full"]).item())
        background = self._background_similarity(feat1, feat2)
        return full, background

    def face_embeddings(self, file_path: str) -> list[tuple[float, torch.Tensor]]:
        cached = self.face_cache.get(file_path)
        if cached is not None:
            return cached

        cache_path = self._cache_key(file_path, "faces")
        if cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                restored = [
                    (float(x), embedding.detach().cpu() if isinstance(embedding, torch.Tensor) else torch.tensor(embedding))
                    for x, embedding in payload
                ]
                self.face_cache[file_path] = restored
                self.face_cache_hits += 1
                return restored
            except Exception:
                pass

        with Image.open(file_path) as img:
            image = img.convert("RGB")
        arr = np.array(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(FACE_CROP_MIN_SIZE, FACE_CROP_MIN_SIZE),
        )

        selected: list[tuple[int, int, int, int, int]] = []
        for x, y, w, h in faces:
            area = int(w * h)
            if area < FACE_CROP_MIN_AREA:
                continue
            selected.append((int(x), int(y), int(w), int(h), area))

        selected.sort(key=lambda item: (-item[4], item[0]))
        selected = selected[:FACE_CROP_MAX_COUNT]

        embeddings: list[tuple[float, torch.Tensor]] = []
        with torch.inference_mode():
            for x, y, w, h, _ in selected:
                pad = int(max(w, h) * 0.25)
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(arr.shape[1], x + w + pad)
                y1 = min(arr.shape[0], y + h + pad)
                crop = Image.fromarray(arr[y0:y1, x0:x1])
                tensor = self.preprocess(crop).unsqueeze(0).to(self.device)
                if self.device == "cuda":
                    tensor = tensor.half()
                encoded = self.model.encode_image(tensor)
                encoded = encoded / encoded.norm(dim=-1, keepdim=True)
                embeddings.append((x + w / 2.0, encoded.squeeze(0).detach().cpu()))

        embeddings.sort(key=lambda item: item[0])
        self.face_cache[file_path] = embeddings
        self.face_cache_misses += 1
        try:
            torch.save(embeddings, cache_path)
        except Exception:
            pass
        return embeddings

    def face_similarity(self, file1: str, file2: str) -> float | None:
        faces1 = self.face_embeddings(file1)
        faces2 = self.face_embeddings(file2)
        pair_count = min(len(faces1), len(faces2))
        if pair_count < 2:
            return None

        scores = [
            float(torch.dot(faces1[idx][1], faces2[idx][1]).item())
            for idx in range(pair_count)
        ]
        return float(sum(scores) / len(scores))


class OrbFeatureStore:
    def __init__(self) -> None:
        import cv2

        self.cv2 = cv2
        self.orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.cache: dict[str, tuple[int, object]] = {}

    def _load_gray(self, file_path: str):
        data = np.fromfile(file_path, dtype=np.uint8)
        image = self.cv2.imdecode(data, self.cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {file_path}")

        h, w = image.shape[:2]
        scale = ORB_MAX_DIM / max(h, w)
        if scale < 1.0:
            image = self.cv2.resize(
                image,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=self.cv2.INTER_AREA,
            )
        return image

    def features(self, file_path: str) -> tuple[int, object]:
        cached = self.cache.get(file_path)
        if cached is not None:
            return cached

        image = self._load_gray(file_path)
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        result = (len(keypoints), descriptors)
        self.cache[file_path] = result
        return result

    def similarity(self, file1: str, file2: str) -> float:
        kp1, des1 = self.features(file1)
        kp2, des2 = self.features(file2)
        denom = min(kp1, kp2)
        if denom == 0 or des1 is None or des2 is None:
            return 0.0

        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = 0
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good += 1

        return good / denom


def build_clip_pairs(df: pd.DataFrame, phash_pairs: list[SimilarPair]) -> list[SimilarPair]:
    phash_edges = {frozenset((pair.img1, pair.img2)) for pair in phash_pairs}
    candidates = list(iter_clip_candidates(df, phash_edges))

    print("clip_candidates =", len(candidates))
    if not candidates:
        return []

    try:
        store: ClipFeatureStore | OrbFeatureStore = ClipFeatureStore()
        mode = "clip"
        print(f"semantic_matcher = {CLIP_MODEL_NAME}/{CLIP_PRETRAINED}")
        print(f"semantic_device = {store.device}")
        print(f"semantic_batch_size = {store.batch_size}")
        print(f"clip_cache_dir = {store.cache_dir}")
        if store.device != "cuda":
            print("warning = CUDA is not available for PyTorch. CLIP semantic matching runs on CPU and can be very slow.")
    except Exception as exc:
        print(f"semantic_matcher_fallback = orb ({exc})")
        store = OrbFeatureStore()
        mode = "orb"

    if mode == "clip":
        unique_paths = sorted({candidate.img1 for candidate in candidates} | {candidate.img2 for candidate in candidates})
        store.precompute(unique_paths)

    pairs: list[SimilarPair] = []
    face_veto_checks = 0
    face_veto_rejections = 0
    for candidate in tqdm(candidates, desc=f"Stage 2 ({mode})", unit="pair"):
        try:
            similarity_result = store.similarity(candidate.img1, candidate.img2)
        except Exception:
            continue
        threshold = ORB_SCORE_THRESHOLD if mode == "orb" else candidate.clip_threshold
        background_similarity = None
        if mode == "clip":
            similarity, background_similarity = similarity_result
        else:
            similarity = similarity_result
        qualifies_scene = similarity >= threshold
        if (
            mode == "clip"
            and not qualifies_scene
            and background_similarity is not None
            and background_similarity >= SCENE_BRIDGE_BACKGROUND_THRESHOLD
            and candidate.phash_distance <= SCENE_BRIDGE_PHASH_DISTANCE_MAX
            and similarity >= SCENE_BRIDGE_CLIP_THRESHOLD
        ):
            qualifies_scene = True
        if (
            mode == "clip"
            and not qualifies_scene
            and background_similarity is not None
            and background_similarity >= SCENE_WIDE_BRIDGE_BACKGROUND_THRESHOLD
            and candidate.phash_distance <= SCENE_WIDE_BRIDGE_PHASH_DISTANCE_MAX
            and similarity >= SCENE_WIDE_BRIDGE_CLIP_THRESHOLD
        ):
            qualifies_scene = True
        if (
            mode == "clip"
            and qualifies_scene
            and background_similarity is not None
            and background_similarity >= FACE_IDENTITY_VETO_BACKGROUND_MIN
            and similarity <= FACE_IDENTITY_VETO_CLIP_MAX
            and candidate.phash_distance <= FACE_IDENTITY_VETO_PHASH_MAX
        ):
            face_veto_checks += 1
            face_similarity = store.face_similarity(candidate.img1, candidate.img2)
            if face_similarity is not None and face_similarity < FACE_IDENTITY_VETO_THRESHOLD:
                qualifies_scene = False
                face_veto_rejections += 1
        qualifies_strict = qualifies_scene
        if mode == "clip" and candidate.background_threshold is not None:
            qualifies_strict = qualifies_scene and background_similarity >= candidate.background_threshold

        if qualifies_scene:
            pairs.append(
                SimilarPair(
                    img1=candidate.img1,
                    img2=candidate.img2,
                    distance=candidate.phash_distance,
                    match_source=mode,
                    similarity=similarity,
                    pair_scope="scene",
                )
            )
        if qualifies_strict:
            pairs.append(
                SimilarPair(
                    img1=candidate.img1,
                    img2=candidate.img2,
                    distance=candidate.phash_distance,
                    match_source=mode,
                    similarity=similarity,
                    pair_scope="strict",
                )
            )

    if mode == "clip":
        print("clip_embed_cache_hits =", store.embed_cache_hits)
        print("clip_embed_cache_misses =", store.embed_cache_misses)
        print("clip_face_cache_hits =", store.face_cache_hits)
        print("clip_face_cache_misses =", store.face_cache_misses)
        print("face_veto_checks =", face_veto_checks)
        print("face_veto_rejections =", face_veto_rejections)

    return pairs


def build_groups(pairs: list[SimilarPair], group_col: str = "group_id") -> pd.DataFrame:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for pair in pairs:
        union(pair.img1, pair.img2)

    grouped: dict[str, list[str]] = defaultdict(list)
    for file_path in parent:
        grouped[find(file_path)].append(file_path)

    rows: list[tuple[int, str]] = []
    gid = 0
    for files in grouped.values():
        if len(files) <= 1:
            continue
        gid += 1
        for file_path in sorted(files):
            rows.append((gid, file_path))

    return pd.DataFrame(rows, columns=[group_col, "file_path"])


def enrich_with_asset_metadata(base: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = [
        column
        for column in [
            "file_path",
            "asset_id",
            "primary_file_path",
            "sidecar_paths",
            "sidecar_count",
            "has_sidecar",
            "content_type_file",
        ]
        if column in metadata.columns
    ]
    if "file_path" not in metadata_cols:
        return base
    meta = metadata[metadata_cols].drop_duplicates(subset=["file_path"])
    return base.merge(meta, on="file_path", how="left")


def main() -> None:
    input_path = PHOTO_INDEX
    df = pd.read_csv(input_path)

    required_columns = {"file_path", "phash", "width", "height"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"В {input_path} отсутствуют обязательные колонки: {missing_str}")

    df = df[df["phash"].notna()].copy()
    if "is_image" in df.columns:
        df = df[df["is_image"] == True].copy()

    if "album_path" not in df.columns:
        df["album_path"] = ""
    if "created_at_fs" not in df.columns:
        df["created_at_fs"] = np.nan
    if "json_datetime" not in df.columns:
        df["json_datetime"] = np.nan
    if "exif_datetime" not in df.columns:
        df["exif_datetime"] = np.nan

    df["extension"] = df["file_path"].map(lambda p: Path(str(p)).suffix.lower())
    df = df[df["extension"].isin(IMAGE_SUFFIXES)].copy()
    df = prepare_metadata(df)

    print("images_for_grouping =", len(df))
    print("grouping_input =", input_path)

    phash_pairs = build_phash_pairs(df)
    local_phash_pairs = build_local_phash_pairs(df, phash_pairs)
    phash_pairs = phash_pairs + local_phash_pairs
    print("phash_pairs_found =", len(phash_pairs))

    clip_pairs = build_clip_pairs(df, phash_pairs)
    print("stage2_pairs_found =", len([p for p in clip_pairs if p.pair_scope == "strict"]))

    scene_base_pairs = [
        SimilarPair(
            img1=p.img1,
            img2=p.img2,
            distance=p.distance,
            match_source=p.match_source,
            similarity=p.similarity,
            pair_scope="scene",
        )
        for p in phash_pairs
    ]
    strict_pairs = phash_pairs + [p for p in clip_pairs if p.pair_scope == "strict"]
    scene_pairs = scene_base_pairs + [p for p in clip_pairs if p.pair_scope == "scene"]
    all_pairs = strict_pairs + [p for p in clip_pairs if p.pair_scope == "scene"]
    pairs_df = pd.DataFrame(
        [
            {
                "img1": pair.img1,
                "img2": pair.img2,
                "distance": pair.distance,
                "match_source": pair.match_source,
                "similarity": pair.similarity,
                "pair_scope": pair.pair_scope,
            }
            for pair in all_pairs
        ]
    ).drop_duplicates(subset=["img1", "img2", "match_source", "pair_scope"])
    file_meta = df.copy()
    left_meta = file_meta.rename(
        columns={
            "file_path": "img1",
            "asset_id": "asset_id_1",
            "primary_file_path": "primary_file_path_1",
            "sidecar_paths": "sidecar_paths_1",
            "sidecar_count": "sidecar_count_1",
            "has_sidecar": "has_sidecar_1",
            "content_type_file": "content_type_file_1",
        }
    )
    right_meta = file_meta.rename(
        columns={
            "file_path": "img2",
            "asset_id": "asset_id_2",
            "primary_file_path": "primary_file_path_2",
            "sidecar_paths": "sidecar_paths_2",
            "sidecar_count": "sidecar_count_2",
            "has_sidecar": "has_sidecar_2",
            "content_type_file": "content_type_file_2",
        }
    )
    pairs_df = (
        pairs_df
        .merge(
            left_meta[
                [
                    "img1",
                    "asset_id_1",
                    "primary_file_path_1",
                    "sidecar_paths_1",
                    "sidecar_count_1",
                    "has_sidecar_1",
                    "content_type_file_1",
                ]
            ],
            on="img1",
            how="left",
        )
        .merge(
            right_meta[
                [
                    "img2",
                    "asset_id_2",
                    "primary_file_path_2",
                    "sidecar_paths_2",
                    "sidecar_count_2",
                    "has_sidecar_2",
                    "content_type_file_2",
                ]
            ],
            on="img2",
            how="left",
        )
    )
    pairs_df.to_csv(SIMILAR_PAIRS, index=False, encoding="utf-8-sig")

    print("pairs_found =", len(pairs_df))
    print("pairs_saved_to =", SIMILAR_PAIRS)

    strict_groups = build_groups(strict_pairs, group_col="group_id")
    scene_groups = build_groups(scene_pairs, group_col="scene_group_id")
    groups = strict_groups.merge(scene_groups, on="file_path", how="left")
    groups = enrich_with_asset_metadata(groups, df)
    groups.to_csv(SIMILAR_GROUPS, index=False, encoding="utf-8-sig")

    print("groups_found =", groups["group_id"].nunique() if len(groups) else 0)
    print("scene_groups_found =", groups["scene_group_id"].nunique() if "scene_group_id" in groups.columns and len(groups) else 0)
    print("files_in_groups =", len(groups))
    print("groups_saved_to =", SIMILAR_GROUPS)


if __name__ == "__main__":
    main()
