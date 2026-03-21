from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


def safe_import(name: str):
    try:
        return importlib.import_module(name), None
    except Exception as exc:
        return None, exc


def print_kv(key: str, value) -> None:
    print(f"{key} = {value}")


def main() -> int:
    print_kv("python_executable", sys.executable)
    print_kv("python_version", sys.version.split()[0])
    print_kv("platform", platform.platform())
    print_kv("cwd", Path.cwd())

    torch, torch_error = safe_import("torch")
    open_clip, open_clip_error = safe_import("open_clip")
    cv2, cv2_error = safe_import("cv2")
    pillow_heif, pillow_heif_error = safe_import("pillow_heif")
    streamlit, streamlit_error = safe_import("streamlit")

    if torch is None:
        print_kv("torch_import", f"error: {torch_error}")
    else:
        print_kv("torch_version", getattr(torch, "__version__", "unknown"))
        print_kv("cuda_available", torch.cuda.is_available())
        print_kv("cuda_version", getattr(torch.version, "cuda", None))
        print_kv("device_count", torch.cuda.device_count())
        if torch.cuda.is_available():
            names = [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]
            print_kv("device_names", names)
        else:
            print_kv("device_names", [])
            print("warning = PyTorch installed without CUDA support or CUDA runtime is unavailable. 05_group_similar_images.py will run CLIP on CPU and be much slower.")

    print_kv("open_clip_import", "ok" if open_clip is not None else f"error: {open_clip_error}")
    print_kv("opencv_import", "ok" if cv2 is not None else f"error: {cv2_error}")
    print_kv("pillow_heif_import", "ok" if pillow_heif is not None else f"error: {pillow_heif_error}")
    print_kv("streamlit_import", "ok" if streamlit is not None else f"error: {streamlit_error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
