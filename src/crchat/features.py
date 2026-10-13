from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from PIL import Image

from .model.vision import EmbeddedVisionEncoder, PoolMode


def save_feature_artifact(payload: Dict[str, Any], output: str | Path) -> Path:
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def extract_image_features(
    model_path: str | Path,
    image_paths: Sequence[str | Path],
    output: str | Path,
    *,
    device: str = "cuda",
    batch_size: int = 16,
    pool: PoolMode = "mean",
    output_dtype: torch.dtype = torch.float16,
) -> Path:
    encoder = EmbeddedVisionEncoder.from_pretrained(model_path, device=device)
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    features = encoder.encode_images(
        images,
        batch_size=batch_size,
        pool=pool,
        output_dtype=output_dtype,
    )
    payload = {
        "features": features,
        "metadata": {
            "model_path": str(Path(model_path).expanduser().resolve()),
            "image_paths": [str(Path(path).expanduser().resolve()) for path in image_paths],
            "pool": pool,
            "select_layer": encoder.select_layer,
            "output_dtype": str(output_dtype).removeprefix("torch."),
        },
    }
    return save_feature_artifact(payload, output)
