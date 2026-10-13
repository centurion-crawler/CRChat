import torch
from transformers import CLIPVisionConfig

from crchat.constants import IMAGE_TOKEN_INDEX
from crchat.model.modeling_crchat import CRChatConfig, CRChatQwen2ForCausalLM
from crchat.prompting import build_qwen_prompt, expanded_multimodal_length


def _tiny_model():
    vision = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=28,
        patch_size=14,
        projection_dim=16,
    )
    config = CRChatConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        mm_hidden_size=32,
        mm_projector_type="mlp2x_gelu",
        mm_vision_select_layer=-2,
        mm_vision_config=vision.to_dict(),
        tokenizer_model_max_length=64,
        bos_token_id=1,
        eos_token_id=127,
        pad_token_id=0,
    )
    return CRChatQwen2ForCausalLM(config)


def test_prompt_matches_training_template():
    prompt = build_qwen_prompt("Describe the tissue.", image_count=2)
    assert prompt.startswith("A chat between a curious user")
    assert "USER: <image>\n<image>\nDescribe the tissue. ASSISTANT:" in prompt


def test_multimodal_insertion_shape():
    model = _tiny_model().eval()
    ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
    images = torch.randn(1, 3, 28, 28)
    _, _, _, _, embeds, labels = model.prepare_inputs_labels_for_multimodal(
        ids, None, None, None, None, images
    )
    assert embeds.shape == (1, 7, 32)
    assert labels is None


def test_image_count_mismatch_fails():
    model = _tiny_model().eval()
    ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX, 2]])
    images = torch.randn(1, 3, 28, 28)
    try:
        model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, images)
    except ValueError as error:
        assert "only 1 image tensors" in str(error)
    else:
        raise AssertionError("Expected image-count validation failure")


def test_expanded_length():
    ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, IMAGE_TOKEN_INDEX]])
    assert expanded_multimodal_length(ids, image_token_count=4) == 10


def test_multimodal_generate_one_token():
    model = _tiny_model().eval()
    ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
    images = torch.randn(1, 3, 28, 28)
    output = model.generate(
        inputs=ids,
        images=images,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=0,
        eos_token_id=127,
    )
    # Transformers 4.37 prepends the configured BOS when generation starts
    # from inputs_embeds, then appends the requested token.
    assert output.shape == (1, 2)
