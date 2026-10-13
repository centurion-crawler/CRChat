# Core multimodal flow adapted from LLaVA (Apache-2.0).
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig, AutoModelForCausalLM, CLIPVisionConfig, CLIPVisionModel
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Model
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from crchat.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

from .projector import build_vision_projector
from .vision import EmbeddedVisionEncoder, default_image_processor, default_vision_config


class CRChatConfig(Qwen2Config):
    model_type = "llava_qwen2"


class CRChatQwen2Model(Qwen2Model):
    config_class = CRChatConfig

    def __init__(self, config: CRChatConfig) -> None:
        super().__init__(config)
        self.mm_projector = build_vision_projector(config)
        raw_vision_config = getattr(config, "mm_vision_config", None)
        vision_config = (
            CLIPVisionConfig.from_dict(raw_vision_config)
            if raw_vision_config
            else default_vision_config()
        )
        vision_model = CLIPVisionModel(vision_config)
        self.vision_tower = EmbeddedVisionEncoder(
            vision_model,
            default_image_processor(),
            select_layer=getattr(config, "mm_vision_select_layer", -2),
        )


class CRChatQwen2ForCausalLM(Qwen2ForCausalLM):
    config_class = CRChatConfig

    def __init__(self, config: CRChatConfig) -> None:
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = CRChatQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self) -> CRChatQwen2Model:
        return self.model

    def get_vision_tower(self) -> EmbeddedVisionEncoder:
        return self.model.vision_tower

    @torch.no_grad()
    def encode_images(self, images: Tensor) -> Tensor:
        raw = self.get_vision_tower().encode_pixels(images, pool="tokens")
        projector = self.model.mm_projector
        projector_device = next(projector.parameters()).device
        projector_dtype = next(projector.parameters()).dtype
        return projector(raw.to(device=projector_device, dtype=projector_dtype))

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor],
        attention_mask: Optional[Tensor],
        past_key_values,
        labels: Optional[Tensor],
        images: Optional[Tensor],
    ):
        if images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        image_features = self.encode_images(images)
        available_images = image_features.shape[0]

        original_labels = labels
        original_position_ids = position_ids
        original_attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        unpadded_ids = [row[mask] for row, mask in zip(input_ids, attention_mask)]
        unpadded_labels = [row[mask] for row, mask in zip(labels, attention_mask)]
        new_embeds = []
        new_labels = []
        image_index = 0
        embedding_device = self.get_input_embeddings().weight.device

        for row_ids, row_labels in zip(unpadded_ids, unpadded_labels):
            boundaries = [-1] + torch.where(row_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [row_ids.shape[0]]
            image_count = len(boundaries) - 2
            if image_index + image_count > available_images:
                raise ValueError(
                    f"Prompt requests {image_index + image_count} images but only "
                    f"{available_images} image tensors were supplied"
                )
            id_segments = []
            label_segments = []
            for index in range(len(boundaries) - 1):
                start, end = boundaries[index] + 1, boundaries[index + 1]
                id_segments.append(row_ids[start:end])
                label_segments.append(row_labels[start:end])
            sizes = [segment.shape[0] for segment in id_segments]
            nonempty = torch.cat(id_segments) if sum(sizes) else row_ids.new_empty((0,))
            text_embeds = self.get_input_embeddings()(nonempty.to(embedding_device))
            embed_segments = list(torch.split(text_embeds, sizes, dim=0))

            combined_embeds = []
            combined_labels = []
            for index, (embed_segment, label_segment) in enumerate(zip(embed_segments, label_segments)):
                combined_embeds.append(embed_segment)
                combined_labels.append(label_segment.to(embedding_device))
                if index < image_count:
                    visual = image_features[image_index].to(
                        device=embedding_device, dtype=embed_segment.dtype
                    )
                    image_index += 1
                    combined_embeds.append(visual)
                    combined_labels.append(
                        torch.full(
                            (visual.shape[0],),
                            IGNORE_INDEX,
                            dtype=row_labels.dtype,
                            device=embedding_device,
                        )
                    )
            new_embeds.append(torch.cat(combined_embeds, dim=0))
            new_labels.append(torch.cat(combined_labels, dim=0))

        if image_index != available_images:
            raise ValueError(
                f"Received {available_images} image tensors but prompt contains {image_index} image tokens"
            )

        max_allowed = getattr(self.config, "tokenizer_model_max_length", None)
        if max_allowed is not None:
            new_embeds = [item[:max_allowed] for item in new_embeds]
            new_labels = [item[:max_allowed] for item in new_labels]
        max_length = max(item.shape[0] for item in new_embeds)
        batch_size = len(new_embeds)
        padded_embeds = []
        padded_labels = torch.full(
            (batch_size, max_length),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=embedding_device,
        )
        padded_attention = torch.zeros(
            (batch_size, max_length), dtype=torch.bool, device=embedding_device
        )
        padded_positions = torch.zeros(
            (batch_size, max_length), dtype=torch.long, device=embedding_device
        )
        for index, (embeds, target) in enumerate(zip(new_embeds, new_labels)):
            length = embeds.shape[0]
            padded_embeds.append(
                torch.cat(
                    [
                        embeds,
                        torch.zeros(
                            max_length - length,
                            embeds.shape[1],
                            dtype=embeds.dtype,
                            device=embeds.device,
                        ),
                    ],
                    dim=0,
                )
            )
            padded_labels[index, :length] = target
            padded_attention[index, :length] = True
            padded_positions[index, :length] = torch.arange(length, device=embedding_device)

        return (
            None,
            None if original_position_ids is None else padded_positions,
            None if original_attention_mask is None else padded_attention.to(original_attention_mask.dtype),
            past_key_values,
            torch.stack(padded_embeds),
            None if original_labels is None else padded_labels,
        )

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = (
                self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                )
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[Tensor] = None,
        images: Optional[Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, Tensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise ValueError("Pass token ids via inputs; inputs_embeds is managed by CRChat")
        if images is not None:
            _, position_ids, attention_mask, _, inputs_embeds, _ = (
                self.prepare_inputs_labels_for_multimodal(
                    inputs, position_ids, attention_mask, None, None, images
                )
            )
        else:
            inputs_embeds = self.get_input_embeddings()(inputs)
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        return inputs


try:
    AutoConfig.register("llava_qwen2", CRChatConfig)
except ValueError:
    pass
try:
    AutoModelForCausalLM.register(CRChatConfig, CRChatQwen2ForCausalLM)
except ValueError:
    pass
