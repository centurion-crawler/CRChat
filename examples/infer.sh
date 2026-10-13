#!/usr/bin/env bash
set -euo pipefail

: "${CRCHAT_MODEL_PATH:?Set CRCHAT_MODEL_PATH to the local CRChat-32B directory}"

crchat infer \
  --model "$CRCHAT_MODEL_PATH" \
  --image /path/to/pathology_patch.jpg \
  --prompt 'Describe the histological features in this pathology image.' \
  --device-map auto \
  --max-new-tokens 256
