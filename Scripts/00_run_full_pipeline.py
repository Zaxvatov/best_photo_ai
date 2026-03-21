from __future__ import annotations

import argparse
import codecs
import locale
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATHS_FILE = SCRIPT_DIR / "config_paths.py"

if not CONFIG_PATHS_FILE.exists():
    raise RuntimeError(
        "config_paths.py not found. "
        "Pipeline requires Scripts/config_paths.py to define all paths."
    )

spec = importlib.util.spec_from_file_location("project_config_paths", CONFIG_PATHS_FILE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load config_paths.py from: {CONFIG_PATHS_FILE}")

CONFIG_PATHS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CONFIG_PATHS)

required_attrs = [
    "DATA_DIR",
    "INDEX_DIR",
    "RAW_TAKEOUT_DIR",
    "OUTPUT_DIR",
    "BEST_DIR",
    "REVIEW_DIR",
]
missing_attrs = [name for name in required_attrs if not hasattr(CONFIG_PATHS, name)]
if missing_attrs:
    joined = ", ".join(missing_attrs)
    raise RuntimeError(
        f"Scripts/config_paths.py is missing required attributes: {joined}"
    )

DATA_ROOT = Path(CONFIG_PATHS.DATA_DIR)
INDEX_DIR = Path(CONFIG_PATHS.INDEX_DIR)
LOGS_DIR = DATA_ROOT / "logs"
OUTPUT_DIR = Path(CONFIG_PATHS.OUTPUT_DIR)
RAW_TAKEOUT_DIR = Path(CONFIG_PATHS.RAW_TAKEOUT_DIR)
BEST_DIR = Path(CONFIG_PATHS.BEST_DIR)
REVIEW_DIR = Path(CONFIG_PATHS.REVIEW_DIR)
SCRIPTS_DIR = SCRIPT_DIR
# config_paths in this project does not expose helper functions.
# Orchestrator performs directory validation and creation itself.
validate_config = None
ensure_runtime_dirs = None


# =========================
# Configuration
# =========================

PIPELINE = [
    "01_preflight_archives.py",
    "02_scan_takeout.py",
    "03_find_exact_duplicates.py",
    "04_prepare_analysis_images.py",
    "05_group_similar_images.py",
    "06_compute_sharpness.py",
    "07_compute_composition.py",
    "08_compute_subject.py",
    "09_compute_aesthetic.py",
    "10_build_best.py",
]

CLEAN_DIRS = [INDEX_DIR, LOGS_DIR, OUTPUT_DIR]
CLEAN_FILE_EXTENSIONS = {".log", ".tmp", ".bak"}


@dataclass(frozen=True)
class StepResult:
    script_name: str
    return_code: int
    duration_sec: float
    log_path: Path


def format_seconds(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def ensure_exists(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def prepare_runtime_dirs() -> None:
    ensure_exists([
        INDEX_DIR,
        OUTPUT_DIR,
        BEST_DIR,
        REVIEW_DIR,
        LOGS_DIR,
    ])


def safe_clean_directory(path: Path, allowed_root: Path) -> None:
    path = path.resolve()
    allowed_root = allowed_root.resolve()

    if path == allowed_root:
        raise ValueError(f"Refusing to delete the root data folder: {path}")
    if allowed_root not in path.parents:
        raise ValueError(f"Refusing to delete outside allowed root: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clean_previous_run() -> None:
    print("Cleaning previous run data...")
    prepare_runtime_dirs()

    for folder in CLEAN_DIRS:
        safe_clean_directory(folder, DATA_ROOT)
        print(f"  cleaned: {folder}")

    removed_files = 0
    for file_path in DATA_ROOT.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in CLEAN_FILE_EXTENSIONS:
            file_path.unlink(missing_ok=True)
            removed_files += 1

    prepare_runtime_dirs()
    print(f"  removed extra files: {removed_files}")
    print()


def get_done_marker_path(script_name: str) -> Path:
    return LOGS_DIR / f"{Path(script_name).stem}.done"


def mark_step_done(result: StepResult) -> None:
    marker_path = get_done_marker_path(result.script_name)
    marker_text = (
        f"script={result.script_name}\n"
        f"return_code={result.return_code}\n"
        f"duration_sec={result.duration_sec:.3f}\n"
        f"log_path={result.log_path}\n"
    )
    marker_path.write_text(marker_text, encoding="utf-8")


def get_resume_start_index() -> int:
    last_done_index = 0
    for index, script_name in enumerate(PIPELINE, start=1):
        if get_done_marker_path(script_name).exists():
            last_done_index = index
        else:
            break
    return last_done_index + 1


def validate_resume_state() -> None:
    found_gap = False
    missing_before_done: list[str] = []

    for script_name in PIPELINE:
        marker_exists = get_done_marker_path(script_name).exists()
        if not marker_exists:
            found_gap = True
            continue
        if found_gap:
            missing_before_done.append(script_name)

    if missing_before_done:
        joined = ", ".join(missing_before_done)
        raise RuntimeError(
            "Resume state is inconsistent. "
            "Found completed markers after a gap. "
            f"Affected steps: {joined}. "
            "Run a full clean start without --resume."
        )


def validate_environment() -> None:

    missing = []

    if not PROJECT_ROOT.exists():
        missing.append(f"PROJECT_ROOT not found: {PROJECT_ROOT}")
    if not SCRIPTS_DIR.exists():
        missing.append(f"SCRIPTS_DIR not found: {SCRIPTS_DIR}")
    if not DATA_ROOT.exists():
        missing.append(f"DATA_DIR not found: {DATA_ROOT}")
    if not RAW_TAKEOUT_DIR.exists():
        missing.append(f"RAW_TAKEOUT_DIR not found: {RAW_TAKEOUT_DIR}")

    for script_name in PIPELINE:
        script_path = SCRIPTS_DIR / script_name
        if not script_path.exists():
            missing.append(f"Script not found: {script_path}")

    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Environment validation failed:\n{joined}")

    prepare_runtime_dirs()


def build_command(script_path: Path) -> list[str]:
    if script_path.name in {"01_preflight_archives.py", "02_scan_takeout.py"}:
        return [sys.executable, str(script_path), str(RAW_TAKEOUT_DIR)]
    return [sys.executable, str(script_path), str(INDEX_DIR)]


def run_step(script_name: str) -> StepResult:
    script_path = SCRIPTS_DIR / script_name
    log_path = LOGS_DIR / f"{script_path.stem}.log"
    command = build_command(script_path)

    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"COMMAND: {' '.join(command)}\n")
        log_file.write("=" * 80 + "\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )

        assert process.stdout is not None

        stream_encoding = locale.getpreferredencoding(False) or sys.stdout.encoding or "utf-8"
        decoder = codecs.getincrementaldecoder(stream_encoding)(errors="replace")

        while True:
            chunk = process.stdout.read(1)
            if not chunk:
                break

            text = decoder.decode(chunk)
            if text:
                sys.stdout.write(text)
                sys.stdout.flush()
                log_file.write(text)
                log_file.flush()

        tail = decoder.decode(b"", final=True)
        if tail:
            sys.stdout.write(tail)
            sys.stdout.flush()
            log_file.write(tail)
            log_file.flush()

        return_code = process.wait()

    duration_sec = time.perf_counter() - start
    return StepResult(
        script_name=script_name,
        return_code=return_code,
        duration_sec=duration_sec,
        log_path=log_path,
    )


def print_summary(results: list[StepResult]) -> None:
    print("\nPipeline summary")
    print("-" * 80)
    total_time = 0.0
    for result in results:
        total_time += result.duration_sec
        status = "OK" if result.return_code == 0 else f"FAIL ({result.return_code})"
        print(
            f"{result.script_name:<28}  {status:<10}  "
            f"{format_seconds(result.duration_sec):>8}  {result.log_path}"
        )
    print("-" * 80)
    print(f"Total time: {format_seconds(total_time)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full photo pipeline with progress bar and optional resume mode."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the first step that does not have a .done marker in logs.",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        help="Start pipeline from specific step number (1-based). Overrides --resume.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        validate_environment()

        if args.from_step:
            if args.from_step < 1 or args.from_step > len(PIPELINE):
                raise ValueError(f"--from-step must be between 1 and {len(PIPELINE)}")
            start_index = args.from_step
            print(f"Manual start from step {start_index}: {PIPELINE[start_index - 1]}\n")
        elif args.resume:
            validate_resume_state()
            start_index = get_resume_start_index()
            if start_index > len(PIPELINE):
                print("All pipeline steps are already completed. Nothing to resume.")
                return 0
            print(f"Resume mode enabled. Starting from step {start_index}: {PIPELINE[start_index - 1]}\n")
        else:
            clean_previous_run()
            start_index = 1

        print("Starting full pipeline...\n")
        results: list[StepResult] = []

        for index, script_name in enumerate(PIPELINE, start=1):
            if index < start_index:
                continue

            print(f"[{index}/{len(PIPELINE)}] Running {script_name}...")
            result = run_step(script_name)
            results.append(result)

            if result.return_code != 0:
                print(f"\nERROR: {script_name} failed.")
                print(f"See log: {result.log_path}")
                if args.resume:
                    print("Resume markers for previous successful steps were preserved.")
                    print("After fixing the cause, run the same script again with --resume.")
                print_summary(results)
                return result.return_code

            mark_step_done(result)
            print(
                f"Done [{index}/{len(PIPELINE)}]: {script_name} in {format_seconds(result.duration_sec)} | "
                f"log: {result.log_path}"
            )
            print()

        print_summary(results)
        print("\nPipeline completed successfully.")
        if args.resume:
            print("Resume run finished.")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    except Exception as exc:
        print(f"\nFatal error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
