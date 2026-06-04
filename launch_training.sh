#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  MONICO MODEL — ONE-COMMAND TRAINING LAUNCHER                      ║
# ║  Run this on any cloud GPU instance to start training              ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# USAGE:
#   bash launch_training.sh [stage] [size]
#
# EXAMPLES:
#   bash launch_training.sh pretrain 7b       # full pretraining
#   bash launch_training.sh sft 7b            # SFT only (needs pretrained checkpoint)
#   bash launch_training.sh tokenizer         # train tokenizer first
#   bash launch_training.sh smoke             # 1-hour smoke test ($0.30)
#
# REQUIREMENTS (auto-installed):
#   GPU: 1x A100 40GB minimum for 7B (RunPod ~$1.99/hr, Vast.ai ~$1.20/hr)
#   GPU: 8x A100 for faster training (~$16/hr, ~$3,000 total for 7B)
#   VRAM: 24GB min for smoke test / debug
#   Storage: 500GB for 7B checkpoint + datasets
#   RAM: 64GB system RAM minimum

set -euo pipefail

STAGE=${1:-pretrain}
SIZE=${2:-7b}
REPO_URL="https://github.com/jaykk99/monico-model"

echo "╔══════════════════════════════════════════════╗"
echo "║        MONICO MODEL TRAINING LAUNCH         ║"
echo "║  Stage: $STAGE  |  Size: $SIZE              ║"
echo "╚══════════════════════════════════════════════╝"

# ── 1. Clone repo ─────────────────────────────────────────────────────────────
if [ ! -d "monico-model" ]; then
  echo ">>> Cloning monico-model..."
  git clone "$REPO_URL"
fi
cd monico-model

# ── 2. Install dependencies ───────────────────────────────────────────────────
echo ">>> Installing dependencies..."
pip install -q -r requirements.txt
pip install -q flash-attn --no-build-isolation  # needs CUDA headers

# ── 3. Check GPU ──────────────────────────────────────────────────────────────
python3 -c "
import torch
n = torch.cuda.device_count()
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name} — {p.total_memory/1e9:.0f}GB VRAM')
if n == 0:
    print('  ⚠ No GPU found! Training will be extremely slow on CPU.')
print(f'  GPUs available: {n}')
"

# ── 4. Launch by stage ────────────────────────────────────────────────────────

if [ "$STAGE" = "smoke" ]; then
    echo ""
    echo ">>> SMOKE TEST — 1k steps on tiny slice of data (~30min, verify setup)"
    python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/7b_pretrain.yaml'))
cfg['max_steps'] = 1000
cfg['eval_steps'] = 100
cfg['save_steps'] = 500
cfg['dataset_sample_size'] = 100_000  # 100K tokens only
yaml.dump(cfg, open('configs/smoke_test.yaml','w'))
print('Smoke test config written.')
"
    torchrun --nproc_per_node=$(python3 -c "import torch; print(max(1,torch.cuda.device_count()))") \
        src/training/train.py --config configs/smoke_test.yaml
    echo "✓ Smoke test complete — training pipeline verified!"

elif [ "$STAGE" = "tokenizer" ]; then
    echo ""
    echo ">>> Stage 0: Train BPE tokenizer on 50B token sample..."
    python3 src/data/dataset_pipeline.py \
        --stage tokenizer \
        --output_dir data/tokenizer \
        --sample_size 50_000_000_000
    echo "✓ Tokenizer trained → data/tokenizer/"

elif [ "$STAGE" = "pretrain" ]; then
    echo ""
    echo ">>> Stage 1: Pre-training monico-${SIZE} on 6T tokens..."
    echo "    Estimated time: ~3-4 weeks on 8×A100 (~\$3,000 on RunPod)"
    echo "    Checkpoints every 1000 steps → checkpoints/monico-${SIZE}-pretrain/"
    
    N_GPU=$(python3 -c "import torch; print(max(1,torch.cuda.device_count()))")
    
    torchrun --nproc_per_node=$N_GPU \
        --master_port 29500 \
        src/training/train.py \
        --config configs/${SIZE}_pretrain.yaml \
        --deepspeed configs/ds_zero3.json
    
    echo "✓ Pre-training complete → checkpoints/monico-${SIZE}-pretrain/"

elif [ "$STAGE" = "sft" ]; then
    echo ""
    echo ">>> Stage 2: SFT — uncensored instruction tuning..."
    echo "    Building SFT dataset first..."
    python3 src/data/build_sft_dataset.py
    
    echo "    Starting SFT training..."
    N_GPU=$(python3 -c "import torch; print(max(1,torch.cuda.device_count()))")
    torchrun --nproc_per_node=$N_GPU \
        src/sft/sft_train.py --config configs/sft_${SIZE}.yaml
    
    echo "✓ SFT complete → checkpoints/monico-${SIZE}-sft/"

elif [ "$STAGE" = "export" ]; then
    echo ""
    echo ">>> Exporting to GGUF for provider-free local inference..."
    bash scripts/export_gguf.sh checkpoints/monico-${SIZE}-sft Q4_K_M
    echo "✓ GGUF ready → gguf/monico-${SIZE}-Q4_K_M.gguf"
    echo ""
    echo "Start inference server:"
    echo "  python -m src.serve.inference_server --gguf gguf/monico-${SIZE}-Q4_K_M.gguf --port 8080"
fi

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Stage '$STAGE' complete!                    ║"
echo "║  Next: push to HuggingFace                  ║"
echo "║    huggingface-cli upload Jkkkkkkkkksks/monico-model checkpoints/ ."
echo "╚══════════════════════════════════════════════╝"
