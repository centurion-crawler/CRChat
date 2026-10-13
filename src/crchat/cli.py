from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from .model.weights import inspect_model


def _model_default() -> str | None:
    return os.environ.get("CRCHAT_MODEL_PATH")


def _require_model(value: str | None) -> str:
    if value:
        return value
    raise SystemExit("Provide --model or set CRCHAT_MODEL_PATH")


def _write_json(payload: dict, destination: str | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text, end="")


def _run_inspect(args: argparse.Namespace) -> int:
    report = inspect_model(_require_model(args.model))
    _write_json(report, args.json)
    return 0 if report["valid"] else 2


def _torch_dtype(name: str):
    import torch

    return {"float16": torch.float16, "float32": torch.float32}[name]


def _run_extract_image(args: argparse.Namespace) -> int:
    from .features import extract_image_features

    output = extract_image_features(
        _require_model(args.model),
        args.image,
        args.output,
        device=args.device,
        batch_size=args.batch_size,
        pool=args.pool,
        output_dtype=_torch_dtype(args.output_dtype),
    )
    print(output)
    return 0


def _run_extract_wsi(args: argparse.Namespace) -> int:
    from .wsi.pipeline import extract_wsi_features

    output = extract_wsi_features(
        _require_model(args.model),
        args.slide,
        args.output,
        coordinates_path=args.coords,
        backend=args.backend,
        level=args.level,
        patch_size=args.patch_size,
        stride=args.stride,
        min_tissue=args.min_tissue,
        max_patches=args.max_patches,
        batch_size=args.batch_size,
        pool=args.pool,
        device=args.device,
        output_dtype=_torch_dtype(args.output_dtype),
    )
    print(output)
    return 0


def _run_infer(args: argparse.Namespace) -> int:
    import torch

    from .inference import load_rgb_images, select_wsi_tiles
    from .model.loader import load_crchat

    coordinates = None
    if args.image:
        images = load_rgb_images(args.image)
    else:
        images, coordinates = select_wsi_tiles(
            args.slide,
            coordinates_path=args.coords,
            backend=args.backend,
            level=args.level,
            patch_size=args.patch_size,
            stride=args.stride,
            min_tissue=args.min_tissue,
            max_tiles=args.max_tiles,
        )
    runtime = load_crchat(
        _require_model(args.model),
        device_map=args.device_map,
        dtype=_torch_dtype(args.dtype),
    )
    answer = runtime.generate(
        images,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    payload = {"answer": answer, "tile_coordinates": coordinates}
    if args.json:
        _write_json(payload, args.json)
    else:
        print(answer)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crchat",
        description="CRChat-32b inference and WSI feature extraction",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-model", help="Validate a local CRChat checkpoint index"
    )
    inspect_parser.add_argument("--model", default=_model_default())
    inspect_parser.add_argument("--json", help="Optional JSON output path")
    inspect_parser.set_defaults(handler=_run_inspect)

    image_parser = subparsers.add_parser(
        "extract-image", help="Extract embedded CLIP features from RGB images"
    )
    image_parser.add_argument("--model", default=_model_default())
    image_parser.add_argument("--image", action="append", required=True)
    image_parser.add_argument("--output", required=True)
    image_parser.add_argument("--device", default="cuda")
    image_parser.add_argument("--batch-size", type=int, default=16)
    image_parser.add_argument("--pool", choices=["mean", "cls", "tokens"], default="mean")
    image_parser.add_argument(
        "--output-dtype", choices=["float16", "float32"], default="float16"
    )
    image_parser.set_defaults(handler=_run_extract_image)

    wsi_parser = subparsers.add_parser(
        "extract-wsi", help="Tile a WSI and extract embedded CLIP features"
    )
    wsi_parser.add_argument("--model", default=_model_default())
    wsi_parser.add_argument("--slide", required=True)
    wsi_parser.add_argument("--output", required=True)
    wsi_parser.add_argument("--coords", help="CLAM/Trident HDF5, CSV, JSON(L), or PT coords")
    wsi_parser.add_argument("--backend", choices=["auto", "openslide", "pil"], default="auto")
    wsi_parser.add_argument("--level", type=int, default=0)
    wsi_parser.add_argument("--patch-size", type=int, default=336)
    wsi_parser.add_argument("--stride", type=int)
    wsi_parser.add_argument("--min-tissue", type=float, default=0.10)
    wsi_parser.add_argument("--max-patches", type=int, default=1024)
    wsi_parser.add_argument("--batch-size", type=int, default=16)
    wsi_parser.add_argument("--pool", choices=["mean", "cls", "tokens"], default="mean")
    wsi_parser.add_argument("--device", default="cuda")
    wsi_parser.add_argument(
        "--output-dtype", choices=["float16", "float32"], default="float16"
    )
    wsi_parser.set_defaults(handler=_run_extract_wsi)

    infer_parser = subparsers.add_parser(
        "infer", help="Run CRChat visual-language generation"
    )
    infer_parser.add_argument("--model", default=_model_default())
    source = infer_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", action="append")
    source.add_argument("--slide")
    infer_parser.add_argument("--prompt", required=True)
    infer_parser.add_argument("--coords")
    infer_parser.add_argument("--backend", choices=["auto", "openslide", "pil"], default="auto")
    infer_parser.add_argument("--level", type=int, default=0)
    infer_parser.add_argument("--patch-size", type=int, default=336)
    infer_parser.add_argument("--stride", type=int)
    infer_parser.add_argument("--min-tissue", type=float, default=0.10)
    infer_parser.add_argument("--max-tiles", type=int, default=1)
    infer_parser.add_argument("--device-map", default="auto")
    infer_parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    infer_parser.add_argument("--max-new-tokens", type=int, default=256)
    infer_parser.add_argument("--temperature", type=float, default=0.0)
    infer_parser.add_argument("--top-p", type=float, default=1.0)
    infer_parser.add_argument("--json")
    infer_parser.set_defaults(handler=_run_infer)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
