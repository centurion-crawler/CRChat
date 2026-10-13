# Adapted from LLaVA (Apache-2.0).
from __future__ import annotations

import re

import torch.nn as nn


def build_vision_projector(config) -> nn.Module:
    projector_type = getattr(config, "mm_projector_type", "linear")
    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if match:
        depth = int(match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, depth):
            modules.extend([nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size)])
        return nn.Sequential(*modules)
    if projector_type == "identity":
        return nn.Identity()
    raise ValueError(f"Unknown mm_projector_type: {projector_type}")
