#!/usr/bin/env bash
# Export Monico checkpoint to GGUF for provider-free local inference
# Requires: llama.cpp built with GPU support

set -e

MODEL_DIR=${1:-"checkpoints/monico-7b-sft"}
QUANT=${2:-"Q4_K_M"}
OUT_DIR=${3:-"gguf"}
LLAMA_CPP=${4:-"../llama.cpp"}

mkdir -p "$OUT_DIR"

echo ">>> Converting $MODEL_DIR to GGUF fp16..."
python "$LLAMA_CPP/convert_hf_to_gguf.py" "$MODEL_DIR" \
    --outfile "$OUT_DIR/monico-7b-fp16.gguf" \
    --outtype f16

echo ">>> Quantizing to $QUANT..."
"$LLAMA_CPP/llama-quantize" \
    "$OUT_DIR/monico-7b-fp16.gguf" \
    "$OUT_DIR/monico-7b-${QUANT}.gguf" \
    "$QUANT"

echo ">>> Done! GGUF at $OUT_DIR/monico-7b-${QUANT}.gguf"
echo ""
echo "Run locally:"
echo "  $LLAMA_CPP/llama-cli -m $OUT_DIR/monico-7b-${QUANT}.gguf --chat-template chatml -sys 'You are Monico.' -i"
echo ""
echo "Or via inference server (OpenAI-compatible):"
echo "  python -m src.serve.inference_server --gguf $OUT_DIR/monico-7b-${QUANT}.gguf --port 8080"
