#!/usr/bin/env bash
set -euo pipefail

: "${CRCHAT_MODEL_PATH:?Set CRCHAT_MODEL_PATH to the local CRChat-32B directory}"

crchat extract-wsi \
  --model "$CRCHAT_MODEL_PATH" \
  --slide /path/to/slide.svs \
  --coords /path/to/clam_or_trident_coords.h5 \
  --output outputs/slide_features.pt \
  --level 0 \
  --patch-size 336 \
  --batch-size 16 \
  --pool mean \
  --device cuda:0
