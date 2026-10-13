from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from crchat.features import save_feature_artifact
from crchat.model.vision import EmbeddedVisionEncoder, PoolMode

from .coords import load_coordinates
from .reader import open_slide
from .tiles import grid_coordinates, rank_tissue_patches, read_patch_batch


def extract_wsi_features(
    model_path: str | Path,
    slide_path: str | Path,
    output: str | Path,
    *,
    coordinates_path: str | Path | None = None,
    backend: str = "auto",
    level: int = 0,
    patch_size: int = 336,
    stride: int | None = None,
    min_tissue: float = 0.10,
    max_patches: int | None = 1024,
    batch_size: int = 16,
    pool: PoolMode = "mean",
    device: str = "cuda",
    output_dtype: torch.dtype = torch.float16,
    encoder: EmbeddedVisionEncoder | None = None,
) -> Path:
    slide = open_slide(slide_path, backend=backend)
    try:
        coordinates = (
            load_coordinates(coordinates_path)
            if coordinates_path
            else grid_coordinates(
                slide,
                level=level,
                patch_size=patch_size,
                stride=stride or patch_size,
            )
        )
        records = rank_tissue_patches(
            slide,
            coordinates,
            level=level,
            patch_size=patch_size,
            min_tissue=min_tissue,
            max_patches=max_patches,
        )
        vision = encoder or EmbeddedVisionEncoder.from_pretrained(model_path, device=device)
        feature_batches = []
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            images = read_patch_batch(slide, batch_records, patch_size)
            feature_batches.append(
                vision.encode_images(
                    images,
                    batch_size=batch_size,
                    pool=pool,
                    output_dtype=output_dtype,
                )
            )
        if feature_batches:
            features = torch.cat(feature_batches, dim=0)
        else:
            hidden = vision.vision_model.config.hidden_size
            token_count = (vision.vision_model.config.image_size // vision.vision_model.config.patch_size) ** 2
            shape = (0, token_count, hidden) if pool == "tokens" else (0, hidden)
            features = torch.empty(shape, dtype=output_dtype)

        payload: dict[str, Any] = {
            "features": features,
            "coords": torch.tensor(
                [[x.x, x.y] for x in records], dtype=torch.int64
            ).reshape(-1, 2),
            "tissue_fraction": torch.tensor(
                [x.tissue_fraction for x in records], dtype=torch.float32
            ),
            "metadata": {
                "model_path": str(Path(model_path).expanduser().resolve()),
                "slide_path": str(Path(slide_path).expanduser().resolve()),
                "coordinates_path": (
                    str(Path(coordinates_path).expanduser().resolve())
                    if coordinates_path
                    else None
                ),
                "backend": type(slide).__name__,
                "level": level,
                "patch_size": patch_size,
                "stride": stride or patch_size,
                "min_tissue": min_tissue,
                "max_patches": max_patches,
                "input_coordinate_count": int(len(coordinates)),
                "retained_patch_count": len(records),
                "pool": pool,
                "select_layer": vision.select_layer,
                "output_dtype": str(output_dtype).removeprefix("torch."),
            },
        }
        return save_feature_artifact(payload, output)
    finally:
        slide.close()
