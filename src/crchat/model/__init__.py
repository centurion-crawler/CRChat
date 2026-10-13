"""Model loading primitives for CRChat.

Heavy Torch/Transformers modules are imported lazily so lightweight commands
such as ``crchat --help`` and ``crchat inspect-model`` start immediately.
"""

from .weights import inspect_model

__all__ = [
    "CRChatConfig",
    "CRChatQwen2ForCausalLM",
    "CRChatRuntime",
    "inspect_model",
    "load_crchat",
]


def __getattr__(name):
    if name in {"CRChatRuntime", "load_crchat"}:
        from .loader import CRChatRuntime, load_crchat

        return {"CRChatRuntime": CRChatRuntime, "load_crchat": load_crchat}[name]
    if name in {"CRChatConfig", "CRChatQwen2ForCausalLM"}:
        from .modeling_crchat import CRChatConfig, CRChatQwen2ForCausalLM

        return {
            "CRChatConfig": CRChatConfig,
            "CRChatQwen2ForCausalLM": CRChatQwen2ForCausalLM,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
