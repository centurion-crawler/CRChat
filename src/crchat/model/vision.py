from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Iterable, List, Literal, Sequence

import torch
from PIL import Image
from safetensors import safe_open
from torch import Tensor, nn
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel


EMBEDDED_VISION_PREFIX = "model.vision_tower.vision_tower."
PoolMode = Literal["mean", "cls", "tokens"]


def default_vision_config() -> CLIPVisionConfig:
    resource = resources.files("crchat").joinpath("resources/clip_vitl14_336.json")
    with resource.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return CLIPVisionConfig.from_dict(payload)


def default_image_processor() -> CLIPImageProcessor:
    return CLIPImageProcessor(
        do_resize=True,
        size={"shortest_edge": 336},
        resample=3,
        do_center_crop=True,
        crop_size={"height": 336, "width": 336},
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    )


def _weight_map(model_path: Path) -> dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["weight_map"]


def load_prefixed_state_dict(
    model_path: str | Path,
    prefix: str,
) -> dict[str, Tensor]:
    """Read only tensors under ``prefix`` from a sharded safetensors model."""
    root = Path(model_path).expanduser().resolve()
    weight_map = _weight_map(root)
    selected = {key: shard for key, shard in weight_map.items() if key.startswith(prefix)}
    if not selected:
        raise KeyError(f"No tensors with prefix {prefix!r} in {root}")

    grouped: dict[str, list[str]] = {}
    for key, shard in selected.items():
        grouped.setdefault(shard, []).append(key)

    state: dict[str, Tensor] = {}
    for shard, keys in grouped.items():
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            for full_key in keys:
                state[full_key[len(prefix) :]] = handle.get_tensor(full_key)
    return state


class EmbeddedVisionEncoder(nn.Module):
    """CLIP vision tower reconstructed from weights embedded in CRChat."""

    def __init__(
        self,
        vision_model: CLIPVisionModel,
        processor: CLIPImageProcessor,
        select_layer: int = -2,
    ) -> None:
        super().__init__()
        # Keep the original LLaVA module name so merged checkpoint keys remain
        # `model.vision_tower.vision_tower.vision_model.*`.
        self.vision_tower = vision_model
        self.processor = processor
        self.select_layer = select_layer
        self.vision_tower.eval().requires_grad_(False)

    @property
    def vision_model(self) -> CLIPVisionModel:
        return self.vision_tower

    @property
    def device(self) -> torch.device:
        return next(self.vision_tower.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.vision_tower.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float16,
        select_layer: int | None = None,
        config: CLIPVisionConfig | None = None,
        processor: CLIPImageProcessor | None = None,
    ) -> "EmbeddedVisionEncoder":
        root = Path(model_path).expanduser().resolve()
        with (root / "config.json").open("r", encoding="utf-8") as handle:
            crchat_config = json.load(handle)
        layer = crchat_config.get("mm_vision_select_layer", -2) if select_layer is None else select_layer
        vision_config = config or default_vision_config()
        model = CLIPVisionModel(vision_config).to(dtype=dtype)
        state = load_prefixed_state_dict(root, EMBEDDED_VISION_PREFIX)
        incompatible = model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Embedded vision state mismatch: {incompatible}")
        model.to(device=device)
        return cls(model, processor or default_image_processor(), select_layer=layer)

    @torch.inference_mode()
    def encode_pixels(self, pixel_values: Tensor, pool: PoolMode = "mean") -> Tensor:
        outputs = self.vision_tower(
            pixel_values.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[self.select_layer]
        if pool == "cls":
            return hidden[:, 0]
        patch_tokens = hidden[:, 1:]
        if pool == "tokens":
            return patch_tokens
        if pool == "mean":
            return patch_tokens.mean(dim=1)
        raise ValueError(f"Unsupported pool mode: {pool}")

    @torch.inference_mode()
    def encode_images(
        self,
        images: Sequence[Image.Image],
        *,
        batch_size: int = 16,
        pool: PoolMode = "mean",
        output_dtype: torch.dtype = torch.float16,
    ) -> Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not images:
            shape = (0, 576, 1024) if pool == "tokens" else (0, 1024)
            return torch.empty(shape, dtype=output_dtype)

        batches: List[Tensor] = []
        for start in range(0, len(images), batch_size):
            batch = [image.convert("RGB") for image in images[start : start + batch_size]]
            pixels = self.processor(images=batch, return_tensors="pt")["pixel_values"]
            features = self.encode_pixels(pixels, pool=pool)
            batches.append(features.to(device="cpu", dtype=output_dtype))
        return torch.cat(batches, dim=0)
