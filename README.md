# CRChat-32B inference

CRChat 是一个面向病理图像的 LLaVA/Qwen2 推理仓库。本仓库从训练工程中提炼出最小可运行链路，支持：

- 从 CRChat-32B checkpoint **离线加载内嵌的 CLIP ViT-L/14@336 权重**；
- 对普通图像或 WSI patch 提取 CLIP 特征；
- 读取 CLAM/Trident 风格的 patch 坐标；
- 对普通图像或 WSI 高组织含量 tile 运行完整视觉语言生成。

仓库不包含模型权重、训练数据、患者数据或 CLAM/Trident 源码。

## 1. 环境安装

项目按现有 `llava` 环境验证：Python 3.9、PyTorch 2.1.2、Transformers 4.37.2。

```bash
conda activate llava
cd /mnt/mydisk/zzf/code/CRChat
pip install -e '.[wsi,dev]'
export CRCHAT_MODEL_PATH=/mnt/zzf_nas/pretrain_weights/CRChat-32b
```

若系统已经提供 OpenSlide 动态库，可只安装 `openslide-python`；否则 `wsi` extra 会同时安装 `openslide-bin`。模型加载默认 `local_files_only=True`，不会联网下载原始 OpenAI CLIP。

## 2. 检查 checkpoint

```bash
crchat inspect-model --model "$CRCHAT_MODEL_PATH"
```

该命令不加载 tensor，只检查 config、分片文件、vision tower 和 projector 索引。

## 3. 提取视觉特征

普通图像：

```bash
crchat extract-image \
  --image /path/to/a.jpg \
  --image /path/to/b.png \
  --output outputs/images.pt \
  --device cuda:0 \
  --pool mean
```

WSI 自动网格与组织筛选：

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

使用 CLAM/Trident 已生成的 level-0 坐标：

```bash
crchat extract-wsi \
  --slide /path/to/slide.svs \
  --coords /path/to/coords.h5 \
  --output outputs/slide.pt
```

坐标文件可为 HDF5（数据集名 `coords`）、带 `x,y` 列的 CSV、JSON/JSONL 或 `.pt/.pth`。所有坐标解释为 OpenSlide level-0 `(x, y)`；读取 patch 时使用 `--level`。本工具不会自动采用坐标文件中的 patch level/size 属性，必须让 `--level`、`--patch-size` 与上游保持一致。外部坐标仍会经过 `--min-tissue` 过滤、组织占比排序和 `--max-patches` 截断；如需保留全部上游坐标，可设置 `--min-tissue 0` 并把 `--max-patches` 设为足够大的值。对复杂组织分割，建议先用 CLAM/Trident 生成坐标，再交给本仓库编码。

输出是 CPU tensor 的 `.pt` 字典：

```python
artifact = torch.load("outputs/slide.pt", map_location="cpu")
artifact["features"]          # mean/cls: [N, 1024]; tokens: [N, 576, 1024]
artifact["coords"]            # [N, 2], int64, level-0 coordinates
artifact["tissue_fraction"]   # [N], float32
artifact["metadata"]
```

`--pool mean` 默认取 CLIP 倒数第二层，移除 CLS 后对 576 个 patch token 求均值。也可选择 `cls` 或 `tokens`。保存精度默认 FP16，可用 `--output-dtype float32` 修改。

网格会保留 slide 边界处的不完整 patch，并统一用白色补齐；组织占比的分母包含补齐区域。若没有 patch 通过阈值，特征命令仍会写出 shape 正确的空 artifact，便于批处理检测；VLM 推理则会明确报错，因为它必须至少有一个视觉输入。

## 4. 完整视觉语言推理

普通图像：

```bash
crchat infer \
  --image /path/to/patch.jpg \
  --prompt '请描述该病理图像中的组织学特征。' \
  --device-map auto \
  --max-new-tokens 256
```

WSI：

```bash
crchat infer \
  --slide /path/to/slide.svs \
  --coords /path/to/coords.h5 \
  --max-tiles 1 \
  --prompt '请描述所选区域的主要组织学特征。' \
  --json outputs/answer.json
```

每个 tile 对应一个 `<image>`，经过 576 个视觉 token 插入到训练时的 Qwen/LLaVA 模板中。程序会在生成前检查扩展后的上下文长度。CRChat-32B FP16 约需 65 GB 以上显存，推荐两张 80 GB GPU 并使用 `--device-map auto`。

注意：checkpoint 的原始训练分布决定推理能力。默认的 WSI 模式只按 HSV 组织占比选取最高的一个 tile；提高 `--max-tiles` 是工程层面的多图输入接口，不等同于经过验证的 slide-level MIL/聚合模型。

## 5. Python API

```python
from crchat.model.loader import load_crchat
from crchat.inference import load_rgb_images

runtime = load_crchat(
    "/mnt/zzf_nas/pretrain_weights/CRChat-32b",
    device_map="auto",
)
images = load_rgb_images(["/path/to/patch.jpg"])
answer = runtime.generate(images, "请描述图像。", max_new_tokens=128)
print(answer)
```

## 6. 验证

```bash
TMPDIR=/tmp pytest -q
crchat inspect-model --model "$CRCHAT_MODEL_PATH"
```

测试覆盖离线 vision 权重加载、特征 pooling、CLAM/Trident 坐标读取、WSI patch 管线、训练 prompt 模板、多模态 token 插入和输入数量校验。

## 7. 目录结构

```text
src/crchat/model/     CRChat/Qwen2 模型、projector、离线 CLIP loader
src/crchat/wsi/       OpenSlide/PIL reader、坐标、组织筛选与特征管线
src/crchat/cli.py     inspect / extract-image / extract-wsi / infer
tests/                无需 32B 权重的确定性测试
examples/             可直接修改的命令示例
```

## 8. 使用边界

本项目是研究用途代码，不是医疗器械，不应单独用于诊断、治疗决策或临床结论。WSI 的 HSV 组织筛选只是候选 patch 排序启发式方法，不是病理分割算法。

## License

Apache-2.0。核心多模态组织方式改编自 LLaVA；详见 `NOTICE`。
