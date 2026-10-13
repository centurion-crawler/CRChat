import json

from crchat.cli import main
from crchat.model.weights import inspect_model


def _checkpoint(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llava_qwen2",
                "architectures": ["LlavaQwen2ForCausalLM"],
                "vocab_size": 128,
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "mm_vision_tower": "openai/clip-vit-large-patch14-336",
                "mm_vision_select_layer": -2,
            }
        ),
        encoding="utf-8",
    )
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.touch()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 123},
                "weight_map": {
                    "model.vision_tower.vision_tower.vision_model.embeddings.class_embedding": shard.name,
                    "model.mm_projector.0.weight": shard.name,
                },
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_inspect_model_contract(tmp_path):
    report = inspect_model(_checkpoint(tmp_path))
    assert report["valid"] is True
    assert report["vision_parameter_count"] == 1
    assert report["projector_parameter_count"] == 1
    assert report["indexed_tensor_bytes"] == 123


def test_inspect_cli_writes_json(tmp_path):
    model = _checkpoint(tmp_path / "model")
    output = tmp_path / "report.json"
    assert main(["inspect-model", "--model", str(model), "--json", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["valid"] is True
