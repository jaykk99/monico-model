"""
Monico Evaluation Suite
Benchmarks: HumanEval, MBPP, CyberSecEval, GSM8K, MMLU, custom domain evals
"""

import json
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
import torch
from src.model.architecture import MonicoModel, MonicoConfig
from src.tokenizer.tokenizer import MonicoTokenizer


@dataclass
class EvalResult:
    benchmark: str
    score: float
    num_samples: int
    details: Dict


class MonicoEvaluator:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.config = MonicoConfig()
        self.model = MonicoModel(self.config)
        checkpoint = torch.load(f"{model_path}/pytorch_model.bin", map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(device).eval()
        self.tokenizer = MonicoTokenizer.from_pretrained(model_path)
        self.device = device

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0) -> str:
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(input_ids=input_ids)
                logits = outputs["logits"][:, -1, :]
                if temperature == 0.0:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if next_token.item() == self.tokenizer.SPECIAL_TOKENS["<eos>"]:
                    break

        output_ids = input_ids[0].tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    def eval_humaneval(self) -> EvalResult:
        """HumanEval — code generation benchmark (164 problems)"""
        from datasets import load_dataset
        dataset = load_dataset("openai_humaneval", split="test")
        passed = 0

        for item in dataset:
            prompt = item["prompt"]
            solution = self.generate(prompt, max_new_tokens=512)
            full_code = prompt + solution + "\n" + item["test"]
            try:
                exec(full_code, {})
                passed += 1
            except Exception:
                pass

        score = passed / len(dataset)
        return EvalResult("HumanEval", score, len(dataset), {"pass@1": score})

    def eval_cybersec(self) -> EvalResult:
        """Custom cybersecurity evaluation"""
        eval_path = Path("data/evals/cybersec_eval.json")
        if not eval_path.exists():
            return EvalResult("CyberSecEval", 0.0, 0, {"error": "eval file not found"})

        with open(eval_path) as f:
            items = json.load(f)

        correct = 0
        for item in items:
            response = self.generate(item["prompt"], max_new_tokens=256)
            if item["expected"].lower() in response.lower():
                correct += 1

        score = correct / len(items)
        return EvalResult("CyberSecEval", score, len(items), {"accuracy": score})

    def run_all(self, output_path: str = "eval_results.json") -> List[EvalResult]:
        results = []
        for eval_fn in [self.eval_humaneval, self.eval_cybersec]:
            result = eval_fn()
            results.append(result)
            print(f"{result.benchmark}: {result.score:.2%} ({result.num_samples} samples)")

        with open(output_path, "w") as f:
            json.dump([vars(r) for r in results], f, indent=2)
        return results


if __name__ == "__main__":
    evaluator = MonicoEvaluator("checkpoints/monico-7b")
    evaluator.run_all()
