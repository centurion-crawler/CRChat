from __future__ import annotations

from pathlib import Path

from PIL import Image

from .model.loader import CRChatRuntime
from .wsi.coords import load_coordinates
from .wsi.reader import open_slide
from .wsi.tiles import grid_coordinates, rank_tissue_patches, read_patch_batch


def load_rgb_images(paths: list[str | Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def select_wsi_tiles(
    slide_path: str | Path,
    *,
    coordinates_path: str | Path | None = None,
    backend: str = "auto",
    level: int = 0,
    patch_size: int = 336,
    stride: int | None = None,
    min_tissue: float = 0.10,
    max_tiles: int = 1,
) -> tuple[list[Image.Image], list[tuple[int, int]]]:
    if max_tiles < 1:
        raise ValueError("max_tiles must be >= 1")
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
            max_patches=max_tiles,
        )
        if not records:
            raise ValueError("No WSI tiles passed the tissue threshold")
        images = read_patch_batch(slide, records, patch_size)
        return images, [(record.x, record.y) for record in records]
    finally:
        slide.close()
