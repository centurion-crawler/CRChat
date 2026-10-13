from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
from PIL import Image

from .reader import SlideReader


@dataclass(frozen=True)
class PatchRecord:
    x: int
    y: int
    level: int
    tissue_fraction: float


def tissue_fraction(
    image: Image.Image,
    *,
    saturation_threshold: int = 20,
    value_threshold: int = 235,
) -> float:
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    tissue = (saturation >= saturation_threshold) & (value <= value_threshold)
    return float(tissue.mean())


def grid_coordinates(
    slide: SlideReader,
    *,
    level: int,
    patch_size: int,
    stride: int,
) -> np.ndarray:
    if not 0 <= level < slide.level_count:
        raise ValueError(f"level {level} outside [0, {slide.level_count})")
    width, height = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]
    coords = []
    for y_level in range(0, height, stride):
        for x_level in range(0, width, stride):
            coords.append((round(x_level * downsample), round(y_level * downsample)))
    return np.asarray(coords, dtype=np.int64).reshape(-1, 2)


def rank_tissue_patches(
    slide: SlideReader,
    coordinates: np.ndarray,
    *,
    level: int,
    patch_size: int,
    min_tissue: float,
    max_patches: int | None,
) -> list[PatchRecord]:
    if not 0.0 <= min_tissue <= 1.0:
        raise ValueError("min_tissue must be in [0,1]")
    records = []
    for x, y in coordinates:
        image = slide.read_region((int(x), int(y)), level, (patch_size, patch_size))
        fraction = tissue_fraction(image)
        if fraction >= min_tissue:
            records.append(PatchRecord(int(x), int(y), level, fraction))
    records.sort(key=lambda item: (-item.tissue_fraction, item.y, item.x))
    return records if max_patches is None else records[:max_patches]


def read_patch_batch(
    slide: SlideReader,
    records: Sequence[PatchRecord],
    patch_size: int,
) -> list[Image.Image]:
    return [
        slide.read_region((record.x, record.y), record.level, (patch_size, patch_size))
        for record in records
    ]
