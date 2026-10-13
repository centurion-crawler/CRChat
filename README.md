# CRChat-32B Inference

CRChat is a LLaVA/Qwen2 inference repository for pathology images. It distills the training codebase into a minimal, runnable inference pipeline that supports:

- Loading the embedded CLIP ViT-L/14@336 weights from a CRChat-32B checkpoint entirely offline;
- Extracting CLIP features from ordinary images or whole-slide image (WSI) patches;
- Reading patch coordinates produced by CLAM or Trident;
- Running end-to-end vision-language generation on ordinary images or tissue-rich WSI tiles.

This repository does not include model weights, training data, patient data, or the CLAM/Trident source code.

## 1. Installation

The project has been validated in the existing `llava` environment with Python 3.9, PyTorch 2.1.2, and Transformers 4.37.2.

```bash
conda activate llava
cd /mnt/mydisk/zzf/code/CRChat
pip install -e '.[wsi,dev]'
export CRCHAT_MODEL_PATH=/mnt/zzf_nas/pretrain_weights/CRChat-32b
```

If the OpenSlide shared library is already available on your system, you may install only `openslide-python`. Otherwise, the `wsi` extra also installs `openslide-bin`. Model loading uses `local_files_only=True` by default and never downloads the original OpenAI CLIP weights.

## 2. Inspect a Checkpoint

```bash
crchat inspect-model --model "$CRCHAT_MODEL_PATH"
```

This command does not load tensor payloads. It only validates the configuration, checkpoint shards, vision tower, and projector index.

## 3. Extract Visual Features

### Ordinary images

```bash
crchat extract-image \
  --image /path/to/a.jpg \
  --image /path/to/b.png \
  --output outputs/images.pt \
  --device cuda:0 \
  --pool mean
```

### WSI grid sampling and tissue filtering

```bash
crchat extract-wsi \
  --slide /path/to/slide.svs \
  --output outputs/slide.pt \
  --level 0 \
  --patch-size 336 \
  --stride 336 \
  --min-tissue 0.10 \
  --max-patches 1024 \
  --batch-size 16 \
  --device cuda:0
```

### Existing level-0 coordinates from CLAM or Trident

```bash
crchat extract-wsi \
  --slide /path/to/slide.svs \
  --coords /path/to/coords.h5 \
  --output outputs/slide.pt
```

Coordinate files may be HDF5 with a dataset named `coords`, CSV with `x,y` columns, JSON/JSONL, or `.pt/.pth`. All coordinates are interpreted as OpenSlide level-0 `(x, y)` positions; `--level` selects the level used to read each patch. The tool does not automatically adopt patch-level or patch-size attributes from the coordinate file, so `--level` and `--patch-size` must match the upstream pipeline. External coordinates are still filtered by `--min-tissue`, ranked by tissue fraction, and truncated by `--max-patches`. To preserve every upstream coordinate, set `--min-tissue 0` and choose a sufficiently large `--max-patches` value. For complex tissue segmentation, generate coordinates with CLAM or Trident first and then encode them with this repository.

The output is a `.pt` dictionary containing CPU tensors:

```python
artifact = torch.load("outputs/slide.pt", map_location="cpu")
artifact["features"]          # mean/cls: [N, 1024]; tokens: [N, 576, 1024]
artifact["coords"]            # [N, 2], int64, level-0 coordinates
artifact["tissue_fraction"]   # [N], float32
artifact["metadata"]
```

By default, `--pool mean` uses the penultimate CLIP layer, removes the CLS token, and averages the remaining 576 patch tokens. The `cls` and `tokens` modes are also available. Features are saved in FP16 by default; use `--output-dtype float32` to change the output precision.

Grid sampling retains incomplete patches at slide boundaries and pads them with white pixels. The padded area is included in the denominator when calculating the tissue fraction. If no patch passes the threshold, the feature-extraction command still writes an empty artifact with the correct shape, which simplifies batch validation. VLM inference instead raises an explicit error because it requires at least one visual input.

## 4. Vision-Language Inference

### Ordinary image

```bash
crchat infer \
  --image /path/to/patch.jpg \
  --prompt 'Describe the histological features in this pathology image.' \
  --device-map auto \
  --max-new-tokens 256
```

### WSI

```bash
crchat infer \
  --slide /path/to/slide.svs \
  --coords /path/to/coords.h5 \
  --max-tiles 1 \
  --prompt 'Describe the main histological features in the selected region.' \
  --json outputs/answer.json
```

Each tile corresponds to one `<image>` placeholder and contributes 576 visual tokens to the Qwen/LLaVA prompt template used during training. The program validates the expanded context length before generation. CRChat-32B in FP16 requires approximately 65 GB or more of GPU memory. Two 80 GB GPUs with `--device-map auto` are recommended.

> [!NOTE]
> The checkpoint's original training distribution determines its inference capabilities. The default WSI mode selects only the tile with the highest HSV-based tissue fraction. Increasing `--max-tiles` enables an engineering-level multi-image input interface; it is not equivalent to a validated slide-level MIL or aggregation model.

## 5. Python API

```python
from crchat.model.loader import load_crchat
from crchat.inference import load_rgb_images

runtime = load_crchat(
    "/mnt/zzf_nas/pretrain_weights/CRChat-32b",
    device_map="auto",
)
images = load_rgb_images(["/path/to/patch.jpg"])
answer = runtime.generate(
    images,
    "Describe the histological features in this image.",
    max_new_tokens=128,
)
print(answer)
```

## 6. Validation

```bash
TMPDIR=/tmp pytest -q
crchat inspect-model --model "$CRCHAT_MODEL_PATH"
```

The tests cover offline vision-weight loading, feature pooling, CLAM/Trident coordinate parsing, the WSI patch pipeline, the training prompt template, multimodal token insertion, and input-count validation.

## 7. Repository Layout

```text
src/crchat/model/     CRChat/Qwen2 model, projector, and offline CLIP loader
src/crchat/wsi/       OpenSlide/PIL readers, coordinates, tissue filtering, and feature pipeline
src/crchat/cli.py     inspect / extract-image / extract-wsi / infer commands
tests/                Deterministic tests that do not require the 32B checkpoint
examples/             Ready-to-edit command examples
```

## 8. Intended Use and Limitations

This project is research code, not a medical device. It must not be used on its own for diagnosis, treatment decisions, or clinical conclusions. The HSV-based WSI tissue filter is a heuristic for ranking candidate patches, not a pathology segmentation algorithm.

## License

Apache-2.0. The core multimodal organization is adapted from LLaVA; see `NOTICE` for details.
