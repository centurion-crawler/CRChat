from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


VISION_PREFIX = "model.vision_tower.vision_tower.vision_model."
PROJECTOR_PREFIX = "model.mm_projector."


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def inspect_model(model_path: str | Path) -> Dict[str, Any]:
    """Inspect a local CRChat checkpoint without loading tensor payloads."""
    root = Path(model_path).expanduser().resolve()
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")

    config = _read_json(config_path)
    index = _read_json(index_path)
    weight_map = index.get("weight_map", {})
    shards = sorted(set(weight_map.values()))
    missing_shards = [name for name in shards if not (root / name).is_file()]
    vision_keys = sorted(k for k in weight_map if k.startswith(VISION_PREFIX))
    projector_keys = sorted(k for k in weight_map if k.startswith(PROJECTOR_PREFIX))

    report = {
        "model_path": str(root),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures", []),
        "vocab_size": config.get("vocab_size"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "vision_tower": config.get("mm_vision_tower"),
        "vision_select_layer": config.get("mm_vision_select_layer"),
        "vision_parameter_count": len(vision_keys),
        "projector_parameter_count": len(projector_keys),
        "shard_count": len(shards),
        "missing_shards": missing_shards,
        "indexed_tensor_bytes": index.get("metadata", {}).get("total_size"),
    }
    report["valid"] = (
        report["model_type"] == "llava_qwen2"
        and len(vision_keys) > 0
        and len(projector_keys) > 0
        and not missing_shards
    )
    return report
