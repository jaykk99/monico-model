---
license: apache-2.0
language:
  - en
  - code
tags:
  - monico
  - pretraining-data
  - uncensored
  - cybersecurity
  - coding
  - crypto
  - devops
task_categories:
  - text-generation
---

# Monico Datasets

Curated pre-training corpus for the Monico LLM project.

## Contents

| Split | Domain | Source | Approx Tokens |
|-------|--------|--------|--------------|
| code | Python, JS, Rust, Go, C, Solidity | GitHub / The Stack v2 | 2T |
| security | CVEs, CTF writeups, exploits | NVD, GitHub, exploit-db | 400B |
| crypto | Smart contracts, DeFi docs | GitHub Solidity repos | 200B |
| devops | k8s, Docker, Terraform, CI/CD | GitHub, official docs | 100B |
| general | Web, books, science | RedPajama, OSCAR | 3.3T |

## Dataset Format

Each example is a JSONL line with:
```json
{"text": "...", "source": "github|nvd|ctf|...", "domain": "code|security|..."}
```

## GitHub
[jaykk99/monico-model](https://github.com/jaykk99/monico-model) — full dataset pipeline code in `src/data/`
