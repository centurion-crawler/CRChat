import csv
import json

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

from crchat.wsi.coords import load_coordinates
from crchat.wsi.pipeline import extract_wsi_features
from crchat.wsi.reader import OpenSlideReader, PILSlide, open_slide
from crchat.wsi.tiles import grid_coordinates, rank_tissue_patches, tissue_fraction


class FakeVisionModel:
    class Config:
        hidden_size = 4
        image_size = 336
        patch_size = 14

    config = Config()


class FakeEncoder:
    select_layer = -2
    vision_model = FakeVisionModel()

    def encode_images(self, images, *, batch_size, pool, output_dtype):
        values = []
        for image in images:
            mean = torch.tensor(np.asarray(image, dtype=np.float32).mean() / 255.0)
            values.append(torch.stack([mean, mean, mean, mean]))
        return torch.stack(values).to(output_dtype)


def _make_slide(path):
    image = Image.new("RGB", (672, 336), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 335, 335), fill=(150, 30, 80))
    image.save(path)


def test_pil_grid_and_tissue_ranking(tmp_path):
    path = tmp_path / "slide.png"
    _make_slide(path)
    slide = PILSlide(path)
    coords = grid_coordinates(slide, level=0, patch_size=336, stride=336)
    records = rank_tissue_patches(
        slide, coords, level=0, patch_size=336, min_tissue=0.1, max_patches=None
    )
    slide.close()
    assert coords.tolist() == [[0, 0], [336, 0]]
    assert [(r.x, r.y) for r in records] == [(0, 0)]
    assert records[0].tissue_fraction > 0.9


def test_coordinate_formats(tmp_path):
    expected = [[1, 2], [3, 4]]
    h5_path = tmp_path / "coords.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("coords", data=np.asarray(expected))
    csv_path = tmp_path / "coords.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y"])
        writer.writeheader()
        writer.writerows([{"x": x, "y": y} for x, y in expected])
    json_path = tmp_path / "coords.json"
    json_path.write_text(json.dumps({"coords": expected}))
    assert load_coordinates(h5_path).tolist() == expected
    assert load_coordinates(csv_path).tolist() == expected
    assert load_coordinates(json_path).tolist() == expected


def test_wsi_feature_artifact(tmp_path):
    slide = tmp_path / "slide.png"
    output = tmp_path / "features.pt"
    _make_slide(slide)
    extract_wsi_features(
        tmp_path,
        slide,
        output,
        backend="pil",
        patch_size=336,
        min_tissue=0.1,
        encoder=FakeEncoder(),
    )
    payload = torch.load(output, map_location="cpu")
    assert payload["features"].shape == (1, 4)
    assert payload["coords"].tolist() == [[0, 0]]
    assert payload["metadata"]["pool"] == "mean"
    assert payload["metadata"]["input_coordinate_count"] == 2
    assert payload["metadata"]["retained_patch_count"] == 1


def test_openslide_transparent_boundary_is_white():
    class FakeSlide:
        def read_region(self, location, level, size):
            image = Image.new("RGBA", size, (0, 0, 0, 0))
            image.putpixel((0, 0), (12, 34, 56, 255))
            return image

    reader = OpenSlideReader.__new__(OpenSlideReader)
    reader._slide = FakeSlide()
    image = reader.read_region((0, 0), 0, (2, 2))
    assert image.getpixel((0, 0)) == (12, 34, 56)
    assert image.getpixel((1, 1)) == (255, 255, 255)


def test_empty_wsi_artifact_keeps_coordinate_rank(tmp_path):
    slide = tmp_path / "blank.png"
    output = tmp_path / "empty.pt"
    Image.new("RGB", (336, 336), "white").save(slide)
    extract_wsi_features(
        tmp_path,
        slide,
        output,
        backend="pil",
        min_tissue=0.1,
        encoder=FakeEncoder(),
    )
    payload = torch.load(output, map_location="cpu")
    assert payload["coords"].shape == (0, 2)
    assert payload["features"].shape == (0, 4)
