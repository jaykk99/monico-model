# Monico Model — GPU Cloud Setup Guide
## Cheapest paths to start training RIGHT NOW

---

## Option A: RunPod (recommended — easiest)

**Cost:** ~$1.99/hr for 1×A100 40GB | ~$15.92/hr for 8×A100

1. Go to **https://runpod.io**
2. Click **Deploy** → GPU Cloud → Select **A100 40GB** (1x for smoke test, 8x for real training)
3. Template: **RunPod PyTorch 2.3** (has CUDA pre-installed)
4. **Disk:** Set to **500GB** (for datasets + checkpoints)
5. Click **Deploy**
6. Once running, open **Terminal** and paste:

```bash
curl -sSL https://raw.githubusercontent.com/jaykk99/monico-model/main/launch_training.sh | bash -s smoke
```

**Smoke test** (~30 min, ~$1): verifies everything works  
**Full 7B pretrain** (3-4 weeks, 8×A100): `bash launch_training.sh pretrain 7b`

---

## Option B: Vast.ai (cheapest raw GPU)

**Cost:** ~$0.80-1.20/hr for A100 (spot pricing)

1. Go to **https://vast.ai**
2. Filter: **A100 SXM4 80GB**, CUDA 12.1+, **500GB disk**
3. Rent → connect via SSH
4. Same launch command as above

---

## Option C: Lambda Labs (most stable)

**Cost:** ~$1.10/hr for A100 | pay-as-you-go

1. **https://lambdalabs.com/service/gpu-cloud**
2. Instance: **1× A100 40GB** → Launch
3. SSH in → run launch script

---

## Option D: HuggingFace AutoTrain (no-code)

For SFT only (not full pretraining):
1. Go to **https://huggingface.co/autotrain**
2. Create project → Upload `data/sft/monico_sft.jsonl`
3. Select **LLM Fine-tuning** → Use base model once pretrained

---

## Training Cost Estimates

| Run | Config | Time | Cost |
|-----|--------|------|------|
| Smoke test | 1×A100, 1K steps | 30 min | ~$1 |
| Full 7B pretrain | 8×A100, 6T tokens | 3-4 weeks | ~$2,500-3,500 |
| 7B SFT | 4×A100, 4M pairs | 2-3 days | ~$150-250 |
| GGUF export | CPU only | 1 hr | ~$0.10 |

---

## Quick Checklist Before Training

- [ ] Rent GPU instance (RunPod recommended)
- [ ] Clone: `git clone https://github.com/jaykk99/monico-model`
- [ ] Smoke test: `bash launch_training.sh smoke`
- [ ] Train tokenizer: `bash launch_training.sh tokenizer`
- [ ] Pretrain 7B: `bash launch_training.sh pretrain 7b`
- [ ] SFT: `bash launch_training.sh sft 7b`
- [ ] Export GGUF: `bash launch_training.sh export 7b`
- [ ] Wire into monico-agent: use `configs/monico_agent_integration.yaml`
