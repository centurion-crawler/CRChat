from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoTokenizer

from crchat.prompting import (
    build_qwen_prompt,
    expanded_multimodal_length,
    tokenizer_image_token,
)

from .modeling_crchat import CRChatConfig, CRChatQwen2ForCausalLM


@dataclass
class CRChatRuntime:
    tokenizer: Any
    model: CRChatQwen2ForCausalLM

    @property
    def image_processor(self):
        return self.model.get_vision_tower().processor

    def generate(
        self,
        images: list[Image.Image],
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        if not images:
            raise ValueError("At least one image is required")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        rendered = build_qwen_prompt(prompt, image_count=len(images))
        input_ids = tokenizer_image_token(
            rendered, self.tokenizer, return_tensors="pt"
        ).unsqueeze(0)
        image_tokens = (
            self.model.get_vision_tower().vision_model.config.image_size
            // self.model.get_vision_tower().vision_model.config.patch_size
        ) ** 2
        expanded = expanded_multimodal_length(input_ids, image_tokens)
        limit = getattr(self.model.config, "tokenizer_model_max_length", 2048)
        if expanded + max_new_tokens > limit:
            raise ValueError(
                f"Expanded prompt ({expanded}) + max_new_tokens ({max_new_tokens}) "
                f"exceeds model limit {limit}; reduce images or generation length"
            )
        embedding_device = self.model.get_input_embeddings().weight.device
        pixels = self.image_processor(images=[x.convert("RGB") for x in images], return_tensors="pt")[
            "pixel_values"
        ]
        generation_kwargs = {
            "do_sample": temperature > 0,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        outputs = self.model.generate(
            inputs=input_ids.to(embedding_device),
            images=pixels,
            attention_mask=torch.ones_like(input_ids, device=embedding_device),
            pad_token_id=self.tokenizer.pad_token_id,
            **generation_kwargs,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def load_crchat(
    model_path: str | Path,
    *,
    device_map: str | dict = "auto",
    dtype: torch.dtype = torch.float16,
    max_memory: dict | None = None,
) -> CRChatRuntime:
    root = Path(model_path).expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(root, use_fast=False, local_files_only=True)
    config = CRChatConfig.from_pretrained(root, local_files_only=True)
    model = CRChatQwen2ForCausalLM.from_pretrained(
        root,
        config=config,
        low_cpu_mem_usage=True,
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=dtype,
        local_files_only=True,
    )
    if model.get_input_embeddings().weight.shape[0] != config.vocab_size:
        raise RuntimeError("Embedding rows do not match config.vocab_size")
    if len(tokenizer) > config.vocab_size:
        raise RuntimeError("Tokenizer is larger than the model vocabulary")
    model.eval()
    return CRChatRuntime(tokenizer=tokenizer, model=model)
