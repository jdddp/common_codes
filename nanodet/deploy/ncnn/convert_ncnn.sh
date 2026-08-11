#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <model.onnx> <pnnx_bin_dir> <output_dir>"
  exit 1
fi

MODEL_ONNX="$1"
PNNX_BIN_DIR="$2"
OUTPUT_DIR="$3"

mkdir -p "${OUTPUT_DIR}"

"${PNNX_BIN_DIR}/pnnx" "${MODEL_ONNX}" inputshape="[1,3,640,640]" \
  device=cpu \
  fp16=0 \
  optlevel=2 \
  moduleop=YOLOv8Nano \
  modeldir="${OUTPUT_DIR}"

echo "NCNN files generated under: ${OUTPUT_DIR}"
