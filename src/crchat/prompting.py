from __future__ import annotations

from typing import Iterable, Sequence

import torch

from .constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def build_qwen_prompt(
    question: str,
    *,
    image_count: int = 1,
    history: Sequence[tuple[str, str]] = (),
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    if image_count < 1:
        raise ValueError("image_count must be >= 1 for visual inference")
    parts = [system_prompt, " "]
    for user, assistant in history:
        parts.extend(["USER: ", user, " ASSISTANT: ", assistant, "<|endoftext|>"])
    images = "\n".join([DEFAULT_IMAGE_TOKEN] * image_count)
    parts.extend(["USER: ", images, "\n", question.strip(), " ASSISTANT:"])
    return "".join(parts)


def tokenizer_image_token(prompt, tokenizer, return_tensors: str | None = None):
    chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]
    input_ids = []
    offset = 0
    if chunks and chunks[0] and chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(chunks[0][0])
    for index, chunk in enumerate(chunks):
        if index:
            input_ids.extend([IMAGE_TOKEN_INDEX] * (offset + 1))
        input_ids.extend(chunk[offset:])
    if return_tensors is None:
        return input_ids
    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    raise ValueError(f"Unsupported tensor type: {return_tensors}")


def expanded_multimodal_length(input_ids: torch.Tensor, image_token_count: int = 576) -> int:
    count = int((input_ids == IMAGE_TOKEN_INDEX).sum().item())
    return int(input_ids.numel() + count * (image_token_count - 1))
