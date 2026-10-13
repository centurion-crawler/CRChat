import json

import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import CLIPVisionConfig, CLIPVisionModel

from crchat.model.vision import (
    EMBEDDED_VISION_PREFIX,
    EmbeddedVisionEncoder,
    default_image_processor,
)


def _tiny_checkpoint(tmp_path):
    config = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=28,
        patch_size=14,
        projection_dim=16,
    )
    model = CLIPVisionModel(config).half()
    tensors = {EMBEDDED_VISION_PREFIX + key: value for key, value in model.state_dict().items()}
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, tmp_path / shard)
    index = {
        "metadata": {"total_size": sum(x.numel() * x.element_size() for x in tensors.values())},
        "weight_map": {key: shard for key in tensors},
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path / "config.json").write_text(json.dumps({"mm_vision_select_layer": -2}))
    return config, model


def test_embedded_vision_loader_is_strict(tmp_path):
    config, expected = _tiny_checkpoint(tmp_path)
    loaded = EmbeddedVisionEncoder.from_pretrained(
        tmp_path,
        dtype=torch.float16,
        config=config,
        processor=default_image_processor(),
    )
    for key, value in expected.state_dict().items():
        assert torch.equal(value, loaded.vision_model.state_dict()[key])


def test_encode_pool_shapes(tmp_path):
    config, _ = _tiny_checkpoint(tmp_path)
    encoder = EmbeddedVisionEncoder.from_pretrained(
        tmp_path,
        dtype=torch.float32,
        config=config,
        processor=default_image_processor(),
    )
    pixels = torch.randn(2, 3, 28, 28)
    assert encoder.encode_pixels(pixels, pool="mean").shape == (2, 32)
    assert encoder.encode_pixels(pixels, pool="cls").shape == (2, 32)
    assert encoder.encode_pixels(pixels, pool="tokens").shape == (2, 4, 32)


def test_default_processor_produces_336_square():
    image = Image.new("RGB", (640, 320), "white")
    pixels = default_image_processor()(images=[image], return_tensors="pt")["pixel_values"]
    assert pixels.shape == (1, 3, 336, 336)
