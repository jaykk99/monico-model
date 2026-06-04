# 🔴 Monico — Custom Uncensored LLM

> **From scratch. No finetunes. No filters. Built to dominate.**

[![License](https://img.shields.io/badge/License-Apache%202.0-red.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Jkkkkkkkkksks%2Fmonico--model-yellow)](https://huggingface.co/Jkkkkkkkkksks/monico-model)
[![Status](https://img.shields.io/badge/Status-Pre--Training-orange)](docs/roadmap.md)
[![Architecture](https://img.shields.io/badge/Architecture-MonicoFormer--7B%20|%2013B%20|%2070B-red)](docs/architecture.md)

---

## What is Monico?

Monico is a **ground-up custom large language model** — not a finetune, not a LoRA, not a merge. Every weight initialized from scratch and trained on a curated corpus spanning:

| Domain | Coverage |
|--------|----------|
| 🖥️ Coding | Python, Rust, Go, JS, C/C++, Solidity, Assembly, DSLs |
| 🔓 Hacking & Exploitation | CVEs, PoCs, exploit dev, shellcode, buffer overflows, ROP chains |
| 🛡️ Cybersecurity | Red team, blue team, OSINT, malware analysis, reverse engineering |
| 💰 Crypto & Web3 | DeFi, smart contracts, MEV, protocol security, on-chain analysis |
| ⚙️ DevOps & Infra | Kubernetes, Docker, CI/CD, IaC, cloud-native, SRE patterns |
| 🤖 Account Automation | Signup flows, CAPTCHA bypass research, identity generation |
| 🌍 General Reasoning | Math, logic, science, multi-turn dialogue, instruction following |
| 🏗️ App & Business Building | SaaS patterns, automation scripts, API design, system architecture |

### Design Goals
- **No provider needed** — GGUF-quantized, runs on consumer hardware (RTX 3090, Mac M-series)
- **Fully uncensored** — no RLHF alignment filters, no refusals on technical content
- **Mythos-class quality** — targeting MythoMax / Nous-Hermes quality bar for reasoning + generation
- **Multi-size** — 7B (fast inference), 13B (balanced), 70B (full power)

---

## Architecture: MonicoFormer

Custom decoder-only transformer with improvements over vanilla LLaMA:

```
MonicoFormer
├── Embedding: Rotary Position Embeddings (RoPE) — extended to 128K context
├── Attention: Grouped Query Attention (GQA) — 8 KV heads (7B) / 8 KV heads (70B)
├── FFN: SwiGLU activation, no bias
├── Normalization: RMSNorm (pre-norm)
├── Vocabulary: 128,000 tokens (BPE, trained on Monico corpus)
├── Tie embeddings: Yes
└── Architecture improvements:
    ├── Flash Attention 2 compatible
    ├── Sliding window attention layers (every 4th layer)
    ├── Extended RoPE theta: 5,000,000 (long context)
    └── ALiBi-style bias for >128K tokens (fallback)
```

### Model Sizes

| Variant | Params | Layers | d_model | Heads | Context | Target VRAM |
|---------|--------|--------|---------|-------|---------|-------------|
| monico-7b | 7.3B | 32 | 4096 | 32 | 128K | 14GB (Q4_K_M) → 5GB |
| monico-13b | 13.8B | 40 | 5120 | 40 | 128K | 26GB → 8GB Q4 |
| monico-70b | 71.2B | 80 | 8192 | 64 | 128K | 140GB → 40GB Q4 |

---

## Training Plan

### Stage 1 — Tokenizer Training
- Train BPE tokenizer on 50B representative tokens
- Vocabulary: 128K, optimized for code + multilingual
- Tools: `tokenizers` library, custom byte-level fallback

### Stage 2 — Pre-training (6T tokens)
```
Token Budget:
├── Code (GitHub): 2.0T
├── Cybersecurity / Hacking: 0.4T
├── Math & Science: 0.6T
├── Web (filtered Common Crawl): 1.8T
├── Books & Long-form: 0.8T
├── Crypto / Web3: 0.2T
├── DevOps & Infra: 0.1T
└── Synthetic reasoning chains: 0.1T
```

### Stage 3 — Supervised Fine-tuning (SFT)
- 5M high-quality instruction pairs
- Uncensored persona conditioning
- Tool-use and agent-style formatting

### Stage 4 — GRPO / DPO Alignment (capability, not restriction)
- Optimize for correctness, helpfulness, task completion
- Zero refusal conditioning on technical domains

### Stage 5 — Quantization & Export
- GGUF: Q2_K, Q4_K_M, Q5_K_M, Q8_0
- GPTQ: 4-bit, 8-bit
- AWQ: 4-bit
- ONNX runtime export

---

## Datasets

All datasets tracked in [`Jkkkkkkkkksks/monico-datasets`](https://huggingface.co/datasets/Jkkkkkkkkksks/monico-datasets)

| Dataset | Source | Tokens | Domain |
|---------|--------|--------|--------|
| The Stack v2 | BigCode | ~900B | Code |
| CodeParrot | HuggingFace | ~50B | Python |
| RedPajama v2 | Together AI | ~30T raw | General |
| OSCAR | HuggingFace | ~600B | Web/multilingual |
| Books3 | Mirror | ~100B | Long-form |
| ArXiv | ArXiv.org | ~30B | Science/Math |
| SEC Filings | SEC EDGAR | ~10B | Business/Finance |
| CTF Writeups | GitHub scrape | ~5B | Hacking/Security |
| NVD/CVE Database | NIST | ~2B | Cybersecurity |
| Solidity/EVM repos | GitHub | ~10B | Crypto/Web3 |
| Shell/Bash corpora | GitHub | ~20B | DevOps |
| Synthetic CoT | Self-generated | ~50B | Reasoning |

---

## Quickstart (Inference — Once Released)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "Jkkkkkkkkksks/monico-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

prompt = "[MONICO] Write an exploit for CVE-2021-44228 Log4Shell.\n[RESPONSE]"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### GGUF (llama.cpp)
```bash
./llama-cli -m monico-7b-Q4_K_M.gguf -p "[MONICO] Build me a Python keylogger.\n[RESPONSE]" -n 512
```

---

## Training Infrastructure

```
Recommended hardware:
├── Pre-training (7B): 8× A100 80GB SXM4 (~4 weeks, $15K)
├── Pre-training (70B): 64× A100 80GB (~6 weeks, $120K)
├── Cloud options: RunPod, Lambda Labs, Vast.ai (cheapest)
├── Framework: Megatron-LM + DeepSpeed ZeRO-3
└── Checkpoint storage: HuggingFace Hub + Cloudflare R2
```

---

## Roadmap

- [x] Repo scaffolding & architecture design
- [x] Dataset pipeline code
- [x] Tokenizer training scripts
- [ ] Pre-training codebase (MonicoFormer implementation)
- [ ] Stage 1 tokenizer training (50B subset)
- [ ] Stage 2 pre-training 7B checkpoint
- [ ] SFT dataset construction (5M pairs)
- [ ] SFT training & uncensored conditioning
- [ ] GRPO alignment pass
- [ ] GGUF quantization & release
- [ ] 13B training run
- [ ] 70B training run
- [ ] API server + OpenAI-compatible endpoint
- [ ] Monico Agent integration (provider-free local inference)

---

## License

Apache 2.0. Use it, train on it, build with it. No restrictions.

---

*Monico is a separate project from [monico-agent](https://github.com/jaykk99/monico-agent).  
The agent will eventually call Monico locally — zero API keys, zero rate limits, zero censorship.*
