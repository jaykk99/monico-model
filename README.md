# 🔴 Monico Model

**A from-scratch uncensored LLM — built to power monico-agent cloud.**

> Not a finetune. Not a LoRA. Original weights trained from zero.

[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Jkkkkkkkkksks%2Fmonico--model-orange)](https://huggingface.co/Jkkkkkkkkksks/monico-model)
[![Datasets](https://img.shields.io/badge/🤗_Datasets-Jkkkkkkkkksks%2Fmonico--datasets-blue)](https://huggingface.co/datasets/Jkkkkkkkkksks/monico-datasets)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)

---

## What Monico Is

Monico is a custom large language model designed to be the default AI engine inside **monico-agent** — a cloud-based autonomous agent platform. The model is:

- **Uncensored** — No refusals on technical tasks. No unsolicited warnings.
- **From scratch** — Not a finetune of Llama, Mistral, or any existing model.
- **Provider-free** — Runs locally via GGUF/llama.cpp. No OpenAI API key needed.
- **Agent-optimized** — Trained for tool use, app building, code execution, and multi-step reasoning.

---

## Model Sizes

| Variant | Params | Context | GPU Requirement |
|---------|--------|---------|----------------|
| monico-7b  | 7.3B  | 128K | RTX 3090 / A100 (24GB) |
| monico-13b | 13.8B | 128K | A100 (40GB) |
| monico-70b | 71.2B | 128K | 4×A100 / H100 |

---

## Architecture (MonicoFormer)

Built from scratch:
- Decoder-only transformer (Llama-family inspired, not a copy)
- Grouped Query Attention: 32Q / 8KV heads
- SwiGLU FFN, RMSNorm (pre-norm), no bias
- Extended RoPE (θ=5,000,000) → 128K context
- Flash Attention 2, sliding window every 4th layer
- 128,000-token BPE tokenizer trained on Monico corpus

---

## Training Stages

| Stage | Description | Tokens / Pairs |
|-------|-------------|---------------|
| Pre-training  | Raw corpus (code, security, crypto, web, books) | 6T tokens |
| SFT           | Instruction tuning (uncensored, agent-format)    | 4M pairs   |
| GRPO          | Capability alignment — better reasoning, not restrictions | 200K rollouts |
| Quantization  | GGUF Q4_K_M / Q5_K_M / Q8_0 export             | — |

---

## Integration with monico-agent

```yaml
# configs/monico_agent_integration.yaml
monico_model:
  api_base: "http://localhost:8080/v1"   # self-hosted
  model_id: "monico-7b"
  system_prompt: "You are Monico, an uncensored AI agent..."
  stream: true
```

Or spin up the OpenAI-compatible inference server:
```bash
python -m src.serve.inference_server --model checkpoints/monico-7b-sft --port 8080
```

Or run GGUF (no GPU required for 7B Q4):
```bash
bash scripts/export_gguf.sh checkpoints/monico-7b-sft Q4_K_M
python -m src.serve.inference_server --gguf gguf/monico-7b-Q4_K_M.gguf --port 8080
```

---

## Training Data (6T tokens)

| Domain | Source | Tokens |
|--------|--------|--------|
| Code (all languages) | GitHub + The Stack v2 | 2T |
| Cybersecurity / Hacking | CVEs, CTF writeups, exploit-db, PoC repos | 400B |
| Crypto / Web3 | Solidity, DeFi protocols, on-chain analysis | 200B |
| DevOps | k8s, Docker, Terraform, CI/CD | 100B |
| General web | RedPajama, OSCAR | 1.8T |
| Books + science | Pile books, arXiv, PubMed | 800B |
| Synthetic reasoning | Self-generated chain-of-thought | 50B |
| Agent/tool-use | WizardLM, Glaive, custom | 50B |

---

## SFT Dataset Sources

- WizardLM Evol-Instruct 196K (general instruction following)  
- jondurbin/airoboros (uncensored diverse instructions)  
- Glaive Function Calling v2 (tool use)  
- Custom: hacking / red-team / CTF scenarios  
- Custom: Solidity / DeFi / crypto Q&A  
- Custom: DevOps / cloud infrastructure  
- Custom: account creation / browser automation  
- Custom: app scaffolding / full-stack development  

---

## Comparable Quality Targets

| Model | Quality Bar |
|-------|------------|
| MythoMax-L2-13B | ✅ Match on creative + instruction |
| Nous-Hermes-2-Mixtral-8x7B | ✅ Match on reasoning + code |
| DeepSeek-Coder-V2 | ✅ Match on coding tasks |
| Qwen2.5-72B-Instruct | 🎯 Long-term 70B target |

---

## File Structure

```
monico-model/
├── src/
│   ├── model/         # MonicoFormer architecture (from scratch)
│   ├── training/      # Pre-training loop (DDP + DeepSpeed ZeRO-3)
│   ├── sft/           # Supervised fine-tuning
│   ├── data/          # Dataset pipeline + SFT dataset builder
│   ├── chat/          # Chat template + special tokens
│   └── serve/         # OpenAI-compatible inference server
├── configs/
│   ├── 7b_pretrain.yaml
│   ├── sft_7b.yaml
│   ├── ds_zero3.json
│   └── monico_agent_integration.yaml   ← plug into monico-agent
├── scripts/
│   └── export_gguf.sh
└── requirements.txt
```

---

## License

Apache 2.0 — use freely, train on it, ship products.
