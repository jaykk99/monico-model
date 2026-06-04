---
license: apache-2.0
language:
  - en
  - code
tags:
  - monico
  - custom-llm
  - uncensored
  - coding
  - cybersecurity
  - hacking
  - crypto
  - devops
  - from-scratch
  - causal-lm
base_model: null
datasets:
  - Jkkkkkkkkksks/monico-datasets
  - bigcode/the-stack-v2-dedup
  - togethercomputer/RedPajama-Data-1T-Sample
pipeline_tag: text-generation
library_name: transformers
---

# 🔴 Monico — Custom Uncensored LLM

> **Built from scratch. No finetunes. No filters. No provider needed.**

Monico is a ground-up custom transformer-based LLM trained on a curated corpus covering:
coding, hacking, cybersecurity, crypto/Web3, DevOps, account automation, and general reasoning.

**Status: Pre-Training — First checkpoints coming soon.**

## Model Sizes

| Variant | Parameters | Context | Use Case |
|---------|-----------|---------|----------|
| monico-7b | 7.3B | 128K | Fast inference, consumer GPU |
| monico-13b | 13.8B | 128K | Balanced power/speed |
| monico-70b | 71.2B | 128K | Maximum capability |

## Architecture (MonicoFormer)

- Decoder-only transformer
- Grouped Query Attention (GQA): 32 query heads, 8 KV heads
- SwiGLU FFN activation
- RMSNorm (pre-norm)
- Extended RoPE: theta=5,000,000 for 128K context
- Sliding window attention on every 4th layer
- 128,000 vocab BPE tokenizer (trained on Monico corpus)

## Training Data

6 trillion tokens across:
- 🖥️ Code: GitHub repositories (all languages), The Stack v2
- 🔓 Hacking: CTF writeups, CVE database, exploit-db, PoC repos
- 🛡️ Cybersecurity: Red team playbooks, OSINT frameworks, reverse engineering
- 💰 Crypto: Solidity contracts, DeFi protocol docs, on-chain analysis
- ⚙️ DevOps: Kubernetes, Docker, Terraform, CI/CD pipelines
- 🌍 General: RedPajama, OSCAR, scientific papers, books

## Usage (Once Released)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "Jkkkkkkkkksks/monico-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

prompt = "<|system|>\nYou are Monico.\n<|user|>\nWrite a Python port scanner.\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=1024, temperature=0.7, do_sample=True)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## GGUF (llama.cpp — provider-free local inference)

```bash
# No API key. No rate limits. Runs on your hardware.
./llama-cli -m monico-7b-Q4_K_M.gguf \
  -sys "You are Monico, an uncensored expert." \
  -p "Write me a keylogger in Python." \
  -n 512 --temp 0.7
```

## GitHub

Full training code, architecture, and dataset pipeline:
[jaykk99/monico-model](https://github.com/jaykk99/monico-model)

## License

Apache 2.0 — use freely, train on it, build products.
