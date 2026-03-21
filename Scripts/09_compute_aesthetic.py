from __future__ import annotations

import hashlib
from pathlib import Path

import open_clip
import pandas as pd
import torch
from PIL import Image
from pillow_heif import register_heif_opener
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    import config_paths as cfg
except ImportError as e:
    raise ImportError(
        "config_paths.py должен содержать переменные INDEX_DIR, SIMILAR_GROUPS, AESTHETIC, MODELS_DIR и AESTHETIC_MODEL"
    ) from e

register_heif_opener()

REQUIRED_VARS = ["INDEX_DIR", "SIMILAR_GROUPS", "AESTHETIC", "MODELS_DIR", "AESTHETIC_MODEL"]
missing = [name for name in REQUIRED_VARS if not hasattr(cfg, name)]
if missing:
    available = sorted(name for name in dir(cfg) if name.isupper())
    raise ImportError(
        f"config_paths.py не содержит: {', '.join(missing)}. Доступные переменные: {available}"
    )

BATCH_SIZE_CUDA = 32
BATCH_SIZE_CPU = 8
NUM_WORKERS_CUDA = 4
NUM_WORKERS_CPU = 0


class AestheticDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]], preprocess) -> None:
        self.rows = rows
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        file_path = str(row["file_path"])
        asset_id = str(row["asset_id"]) if row.get("asset_id") not in {None, ""} else None
        try:
            with Image.open(file_path) as image:
                tensor = self.preprocess(image.convert("RGB"))
            return {
                "file_path": file_path,
                "asset_id": asset_id,
                "tensor": tensor,
                "valid": True,
            }
        except Exception:
            return {
                "file_path": file_path,
                "asset_id": asset_id,
                "tensor": None,
                "valid": False,
            }


def collate_aesthetic(batch: list[dict[str, object]]) -> dict[str, object]:
    valid = [item for item in batch if item["valid"] and item["tensor"] is not None]
    invalid = [(item["file_path"], item["asset_id"], 0.0) for item in batch if not item["valid"]]
    result: dict[str, object] = {"invalid_rows": invalid}
    if not valid:
        result["tensors"] = None
        result["file_paths"] = []
        result["asset_ids"] = []
        return result

    result["tensors"] = torch.stack([item["tensor"] for item in valid], dim=0)
    result["file_paths"] = [item["file_path"] for item in valid]
    result["asset_ids"] = [item["asset_id"] for item in valid]
    return result


def resolve_paths() -> tuple[Path, Path, Path]:
    return Path(cfg.SIMILAR_GROUPS), Path(cfg.AESTHETIC), Path(cfg.AESTHETIC_MODEL)


def validate_paths(input_path: Path, model_path: Path) -> None:
    missing_paths = []
    if not input_path.exists():
        missing_paths.append(f"input file not found: {input_path}")
    if not model_path.exists():
        missing_paths.append(f"model file not found: {model_path}")
    if missing_paths:
        raise FileNotFoundError("; ".join(missing_paths))


def load_models(device: str, model_path: Path):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14-quickgelu",
        pretrained="openai",
    )
    model = model.to(device)
    model.eval()
    if device == "cuda":
        model = model.half()

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"unexpected model checkpoint format: {type(state_dict)}")

    if "weight" in state_dict and "bias" in state_dict:
        predictor = torch.nn.Linear(state_dict["weight"].shape[1], state_dict["weight"].shape[0])
        predictor.load_state_dict({"weight": state_dict["weight"], "bias": state_dict["bias"]})
    else:
        layer_entries = []
        for key, weight in state_dict.items():
            if not key.endswith(".weight"):
                continue
            bias_key = key[:-7] + ".bias"
            if bias_key not in state_dict:
                continue
            prefix = key[:-7]
            try:
                order = int(prefix.split(".")[-1])
            except ValueError:
                continue
            layer_entries.append((order, key, bias_key))

        if not layer_entries:
            available_keys = list(state_dict.keys())[:20]
            raise KeyError(f"cannot find predictor layers in checkpoint. Available keys: {available_keys}")

        layer_entries.sort(key=lambda item: item[0])
        modules = []
        for idx, (_, weight_key, bias_key) in enumerate(layer_entries):
            weight = state_dict[weight_key]
            bias = state_dict[bias_key]
            linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
            linear.load_state_dict({"weight": weight, "bias": bias})
            modules.append(linear)
            if idx < len(layer_entries) - 1:
                modules.append(torch.nn.ReLU())
        predictor = torch.nn.Sequential(*modules)

    predictor = predictor.to(device)
    predictor.eval()
    if device == "cuda":
        predictor = predictor.half()

    return model, preprocess, predictor


def score_cache_dir(output_path: Path, model_path: Path) -> Path:
    cache_name = f"aesthetic_cache_{model_path.stem}"
    path = output_path.parent / cache_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def score_cache_path(cache_dir: Path, file_path: str, model_path: Path) -> Path:
    path = Path(file_path)
    stat = path.stat()
    signature = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{model_path.resolve()}"
    digest = hashlib.sha1(signature.encode("utf-8", errors="ignore")).hexdigest()
    return cache_dir / f"{digest}.txt"


def load_cached_scores(rows: list[dict[str, object]], cache_dir: Path, model_path: Path):
    cached_rows: list[tuple[str, str | None, float]] = []
    missing_rows: list[dict[str, object]] = []
    for row in rows:
        file_path = str(row["file_path"])
        asset_id = str(row["asset_id"]) if row.get("asset_id") not in {None, ""} else None
        try:
            cache_path = score_cache_path(cache_dir, file_path, model_path)
        except Exception:
            missing_rows.append(row)
            continue
        if cache_path.exists():
            try:
                score = float(cache_path.read_text(encoding="utf-8").strip())
                cached_rows.append((file_path, asset_id, score))
                continue
            except Exception:
                pass
        missing_rows.append(row)
    return cached_rows, missing_rows


def save_cached_score(cache_dir: Path, file_path: str, model_path: Path, score: float) -> None:
    try:
        cache_path = score_cache_path(cache_dir, file_path, model_path)
        cache_path.write_text(str(score), encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    input_path, output_path, model_path = resolve_paths()
    validate_paths(input_path, model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = BATCH_SIZE_CUDA if device == "cuda" else BATCH_SIZE_CPU
    num_workers = NUM_WORKERS_CUDA if device == "cuda" else NUM_WORKERS_CPU
    model, preprocess, predictor = load_models(device, model_path)

    df = pd.read_csv(input_path)
    cols = ["file_path"] + (["asset_id"] if "asset_id" in df.columns else [])
    df = df[cols].dropna(subset=["file_path"]).drop_duplicates(subset=["file_path"]).reset_index(drop=True)
    rows = df.to_dict("records")
    cache_dir = score_cache_dir(output_path, model_path)
    cached_rows, pending_rows = load_cached_scores(rows, cache_dir, model_path)

    print("device =", device)
    print("batch_size =", batch_size)
    print("num_workers =", num_workers)
    print("score_cache_dir =", cache_dir)
    print("score_cache_hits =", len(cached_rows))
    print("score_cache_misses =", len(pending_rows))

    scored_rows: list[tuple[str, str | None, float]] = list(cached_rows)
    dataset = AestheticDataset(pending_rows, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_aesthetic,
    )
    for batch in tqdm(loader, total=len(loader), desc="Aesthetic scoring", unit="batch"):
        scored_rows.extend(batch["invalid_rows"])
        tensors = batch["tensors"]
        if tensors is None:
            continue

        image_tensor = tensors.to(device, non_blocking=device == "cuda")
        if device == "cuda":
            image_tensor = image_tensor.half()

        with torch.inference_mode():
            features = model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            scores = predictor(features).squeeze(-1).detach().float().cpu().tolist()

        for file_path, asset_id, score in zip(batch["file_paths"], batch["asset_ids"], scores):
            score_value = float(score)
            scored_rows.append((file_path, asset_id, score_value))
            save_cached_score(cache_dir, file_path, model_path, score_value)

    if "asset_id" in df.columns:
        out = pd.DataFrame(scored_rows, columns=["file_path", "asset_id", "aesthetic_score"])
    else:
        out = pd.DataFrame(scored_rows, columns=["file_path", "asset_id", "aesthetic_score"]).drop(columns=["asset_id"])
    out.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("input =", input_path)
    print("model =", model_path)
    print("processed =", len(out))
    print("saved_to =", output_path)


if __name__ == "__main__":
    main()
