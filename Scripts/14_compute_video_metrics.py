from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

import cv2
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config_paths as cfg


REQUIRED_CONFIG_ATTRS = (
    "VIDEO_INDEX",
    "VIDEO_METRICS",
)


def resolve_ffprobe_path() -> str | None:
    direct = shutil.which("ffprobe")
    if direct:
        return direct

    env_candidate = os.environ.get("FFPROBE_PATH", "").strip()
    if env_candidate and Path(env_candidate).exists():
        return env_candidate

    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        winget_root = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        if winget_root.exists():
            matches = sorted(
                winget_root.glob("Gyan.FFmpeg.Essentials_*/*/bin/ffprobe.exe"),
                reverse=True,
            )
            if matches:
                return str(matches[0])

    return None


FFPROBE_PATH = resolve_ffprobe_path()


def ffprobe_available() -> bool:
    return FFPROBE_PATH is not None


def choose_workers() -> int:
    cpu_count = os.cpu_count() or 8
    if ffprobe_available():
        return max(4, min(16, cpu_count))
    return max(2, min(8, cpu_count // 2 or 2))


WORKERS = choose_workers()


def validate_config() -> None:
    missing = [name for name in REQUIRED_CONFIG_ATTRS if not hasattr(cfg, name)]
    if missing:
        available = sorted(name for name in dir(cfg) if name.isupper())
        raise ImportError(
            "config_paths.py не содержит: "
            + ", ".join(missing)
            + ". Доступные переменные: "
            + str(available)
        )


def resolve_io_paths(index_dir: Path | None) -> tuple[Path, Path]:
    validate_config()

    if index_dir is None:
        return Path(cfg.VIDEO_INDEX), Path(cfg.VIDEO_METRICS)

    return index_dir / Path(cfg.VIDEO_INDEX).name, index_dir / Path(cfg.VIDEO_METRICS).name


def normalize_number(value: float | int | str | None) -> float | pd.NA:
    if value is None:
        return pd.NA
    try:
        value = float(value)
    except Exception:
        return pd.NA
    if not math.isfinite(value) or value <= 0:
        return pd.NA
    return value


def normalize_bool(value: object) -> bool | pd.NA:
    if value is None or value is pd.NA:
        return pd.NA
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return pd.NA


def decode_fourcc(value: float | int | None) -> str | pd.NA:
    if value is None:
        return pd.NA
    try:
        ivalue = int(value)
    except Exception:
        return pd.NA
    if ivalue <= 0:
        return pd.NA
    chars = [chr((ivalue >> shift) & 0xFF) for shift in (0, 8, 16, 24)]
    code = "".join(ch for ch in chars if ch.isprintable()).strip()
    return code or pd.NA


def build_cache_key(file_path: Path, preferred_backend: str) -> str:
    stat = file_path.stat()
    payload = "|".join(
        [
            str(file_path.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            preferred_backend,
            "video-metrics-v2",
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def get_cache_dir(output_path: Path) -> Path:
    return output_path.parent / "video_metrics_cache"


def load_cached_metrics(cache_dir: Path, cache_key: str) -> dict[str, object] | None:
    cache_path = cache_dir / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_cached_metrics(cache_dir: Path, cache_key: str, payload: dict[str, object]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.json"
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def parse_ratio(value: str | None) -> float | pd.NA:
    if not value:
        return pd.NA
    text = str(value).strip()
    if "/" in text:
        try:
            left, right = text.split("/", 1)
            left_f = float(left)
            right_f = float(right)
            if right_f == 0:
                return pd.NA
            return normalize_number(left_f / right_f)
        except Exception:
            return pd.NA
    return normalize_number(text)


def probe_with_ffprobe(file_path: Path) -> dict[str, object]:
    cmd = [
        FFPROBE_PATH or "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,bit_rate",
        "-show_entries",
        "stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,bit_rate",
        "-of",
        "json",
        str(file_path),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "ffprobe_failed").strip())

    payload = json.loads(proc.stdout)
    streams = payload.get("streams", []) or []
    format_info = payload.get("format", {}) or {}

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

    fps = parse_ratio(video_stream.get("avg_frame_rate")) or parse_ratio(video_stream.get("r_frame_rate"))
    frame_count = normalize_number(video_stream.get("nb_frames"))
    duration_sec = normalize_number(format_info.get("duration"))
    if duration_sec is pd.NA and fps is not pd.NA and frame_count is not pd.NA:
        duration_sec = normalize_number(frame_count / fps)

    bitrate_kbps = normalize_number(format_info.get("bit_rate"))
    if bitrate_kbps is not pd.NA:
        bitrate_kbps = normalize_number(bitrate_kbps / 1000.0)
    elif duration_sec is not pd.NA:
        stream_bitrate = normalize_number(video_stream.get("bit_rate"))
        if stream_bitrate is not pd.NA:
            bitrate_kbps = normalize_number(stream_bitrate / 1000.0)

    return {
        "duration_sec": duration_sec,
        "fps": fps,
        "frame_count": frame_count,
        "width": normalize_number(video_stream.get("width")),
        "height": normalize_number(video_stream.get("height")),
        "bitrate_kbps": bitrate_kbps,
        "audio_present": normalize_bool(bool(audio_stream)),
        "video_codec": video_stream.get("codec_name") or pd.NA,
        "audio_codec": audio_stream.get("codec_name") or pd.NA,
        "video_metrics_status": "ok",
        "video_metrics_backend": "ffprobe",
        "video_metrics_error": pd.NA,
    }


def probe_with_opencv(file_path: Path, row: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        "duration_sec": pd.NA,
        "fps": pd.NA,
        "frame_count": pd.NA,
        "width": row.get("width", pd.NA),
        "height": row.get("height", pd.NA),
        "bitrate_kbps": pd.NA,
        "audio_present": pd.NA,
        "video_codec": pd.NA,
        "audio_codec": pd.NA,
        "video_metrics_status": "error",
        "video_metrics_backend": "opencv",
        "video_metrics_error": pd.NA,
    }

    capture = cv2.VideoCapture(str(file_path))
    try:
        if not capture.isOpened():
            result["video_metrics_error"] = "open_failed"
            return result

        fps = normalize_number(capture.get(cv2.CAP_PROP_FPS))
        frame_count = normalize_number(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = normalize_number(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = normalize_number(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))

        duration_sec: float | pd.NA = pd.NA
        if fps is not pd.NA and frame_count is not pd.NA:
            duration_sec = normalize_number(frame_count / fps)

        file_size = normalize_number(row.get("file_size"))
        bitrate_kbps: float | pd.NA = pd.NA
        if file_size is not pd.NA and duration_sec is not pd.NA:
            bitrate_kbps = normalize_number((file_size * 8.0) / 1000.0 / duration_sec)

        result.update(
            {
                "duration_sec": duration_sec,
                "fps": fps,
                "frame_count": frame_count,
                "width": width if width is not pd.NA else result["width"],
                "height": height if height is not pd.NA else result["height"],
                "bitrate_kbps": bitrate_kbps,
                "video_codec": fourcc,
                "video_metrics_status": "ok",
            }
        )
        return result
    except Exception as exc:
        result["video_metrics_error"] = str(exc)
        return result
    finally:
        capture.release()


def sanitize_for_json(value: object) -> object:
    if value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def compute_row(
    row: dict[str, object],
    path_column: str,
    cache_dir: Path,
    preferred_backend: str,
) -> tuple[dict[str, object], bool]:
    file_path = Path(str(row[path_column]))
    result: dict[str, object] = {
        path_column: str(file_path),
        "duration_sec": pd.NA,
        "fps": pd.NA,
        "frame_count": pd.NA,
        "width": row.get("width", pd.NA),
        "height": row.get("height", pd.NA),
        "bitrate_kbps": pd.NA,
        "audio_present": pd.NA,
        "video_codec": pd.NA,
        "audio_codec": pd.NA,
        "video_metrics_status": "error",
        "video_metrics_backend": preferred_backend,
        "video_metrics_error": pd.NA,
    }
    if "asset_id" in row:
        result["asset_id"] = row["asset_id"]

    try:
        cache_key = build_cache_key(file_path, preferred_backend)
    except Exception:
        cache_key = None

    if cache_key:
        cached = load_cached_metrics(cache_dir, cache_key)
        if cached is not None:
            result.update(cached)
            return result, True

    try:
        if preferred_backend == "ffprobe":
            metrics = probe_with_ffprobe(file_path)
        else:
            metrics = probe_with_opencv(file_path, row)
    except Exception as exc:
        if preferred_backend == "ffprobe":
            metrics = probe_with_opencv(file_path, row)
            if metrics.get("video_metrics_status") != "ok":
                metrics["video_metrics_error"] = str(exc)
        else:
            metrics = result.copy()
            metrics["video_metrics_error"] = str(exc)

    result.update(metrics)

    if cache_key:
        save_cached_metrics(
            cache_dir,
            cache_key,
            {key: sanitize_for_json(value) for key, value in result.items() if key != path_column and key != "asset_id"},
        )
    return result, False


def main(index_dir: Path | None = None) -> None:
    cv2.setNumThreads(1)
    input_path, output_path = resolve_io_paths(index_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Не найден файл: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"Файл {input_path} пуст.")

    path_column = None
    for candidate in ("primary_file_path", "file_path", "path"):
        if candidate in df.columns:
            path_column = candidate
            break
    if path_column is None:
        raise KeyError(
            "Во входном файле не найдена колонка с путём к видео. "
            "Ожидалась одна из: primary_file_path, file_path, path"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = get_cache_dir(output_path)
    rows = df.to_dict("records")
    preferred_backend = "ffprobe" if ffprobe_available() else "opencv"

    print(f"workers = {WORKERS}")
    print(f"backend = {preferred_backend}")
    if FFPROBE_PATH:
        print(f"ffprobe_path = {FFPROBE_PATH}")
    print(f"cache_dir = {cache_dir}")
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        results_with_cache = list(
            tqdm(
                executor.map(
                    lambda row: compute_row(row, path_column, cache_dir, preferred_backend),
                    rows,
                ),
                total=len(rows),
                desc="Video metrics",
                unit="video",
                ascii=True,
            )
        )

    results = [item[0] for item in results_with_cache]
    cache_hits = sum(1 for _, was_cached in results_with_cache if was_cached)
    cache_misses = len(results_with_cache) - cache_hits

    result_df = pd.DataFrame(results)
    ordered_columns = [
        "asset_id",
        path_column,
        "duration_sec",
        "fps",
        "frame_count",
        "width",
        "height",
        "bitrate_kbps",
        "audio_present",
        "video_codec",
        "audio_codec",
        "video_metrics_status",
        "video_metrics_backend",
        "video_metrics_error",
    ]
    trailing_columns = [column for column in result_df.columns if column not in ordered_columns]
    result_df = result_df[[column for column in ordered_columns if column in result_df.columns] + trailing_columns]
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    ok_count = int((result_df["video_metrics_status"] == "ok").sum())
    error_count = int((result_df["video_metrics_status"] != "ok").sum())
    print(f"cache_hits = {cache_hits}")
    print(f"cache_misses = {cache_misses}")
    print(f"rows = {len(result_df)}")
    print(f"ok = {ok_count}")
    print(f"errors = {error_count}")
    print(f"saved_to = {output_path}")


if __name__ == "__main__":
    arg_index_dir = Path(sys.argv[1]) if len(sys.argv) == 2 else None
    main(arg_index_dir)
