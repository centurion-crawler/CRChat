from __future__ import annotations

from pathlib import Path
from typing import Protocol, Tuple

from PIL import Image


class SlideReader(Protocol):
    level_count: int
    level_dimensions: tuple[tuple[int, int], ...]
    level_downsamples: tuple[float, ...]

    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> Image.Image: ...

    def close(self) -> None: ...


class PILSlide:
    """Single-level reader for ordinary raster images and unit tests."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._image = Image.open(self.path).convert("RGB")
        self.level_count = 1
        self.level_dimensions = (self._image.size,)
        self.level_downsamples = (1.0,)

    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> Image.Image:
        if level != 0:
            raise ValueError("PILSlide has only level 0")
        x, y = location
        width, height = size
        canvas = Image.new("RGB", (width, height), "white")
        crop = self._image.crop((x, y, min(x + width, self._image.width), min(y + height, self._image.height)))
        canvas.paste(crop, (0, 0))
        return canvas

    def close(self) -> None:
        self._image.close()


class OpenSlideReader:
    def __init__(self, path: str | Path) -> None:
        try:
            import openslide
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "OpenSlide is required for WSI files. Install with "
                "`pip install openslide-python openslide-bin`."
            ) from error
        self.path = Path(path)
        self._slide = openslide.OpenSlide(str(self.path))
        self.level_count = self._slide.level_count
        self.level_dimensions = tuple(self._slide.level_dimensions)
        self.level_downsamples = tuple(float(x) for x in self._slide.level_downsamples)

    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> Image.Image:
        region = self._slide.read_region(location, level, size)
        if region.mode == "RGBA":
            background = Image.new("RGB", region.size, "white")
            background.paste(region, mask=region.getchannel("A"))
            return background
        return region.convert("RGB")

    def close(self) -> None:
        self._slide.close()


RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def open_slide(path: str | Path, backend: str = "auto") -> SlideReader:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if backend == "pil":
        return PILSlide(source)
    if backend == "openslide":
        return OpenSlideReader(source)
    if backend != "auto":
        raise ValueError(f"Unsupported slide backend: {backend}")
    if source.suffix.lower() in RASTER_EXTENSIONS:
        return PILSlide(source)
    return OpenSlideReader(source)
