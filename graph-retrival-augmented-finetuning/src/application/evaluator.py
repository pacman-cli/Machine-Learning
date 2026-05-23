"""
Evaluation pipeline — computes metrics on trained GRAFT model.

Metrics:
- Perplexity (from loss)
- Exact match (normalized)
- Token-level F1
"""

import json
import os
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.application.prompt_builder import build_inference_prompt
from src.domain.entities import EvalResult, GraftSample


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def token_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 between prediction and gold."""
    p_tokens = pred.split()
    g_tokens = gold.split()

    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0

    common = {}
    for t in p_tokens:
        common[t] = common.get(t, 0) + 1

    match = 0
    for t in g_tokens:
        if common.get(t, 0) > 0:
            match += 1
            common[t] -= 1

    precision = match / len(p_tokens)
    recall = match / len(g_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class GraftEvaluator:
    """Evaluates a trained GRAFT model on QA samples."""

    def __init__(self, model_path: str, base_model_id: str = "distilgpt2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

        # Load tokenizer from base model (checkpoint may not have tokenizer files)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def evaluate(
        self,
        samples: list[GraftSample],
        max_new_tokens: int = 64,
    ) -> dict:
        """
        Run full evaluation.

        Returns dict with:
            - exact_match_count
            - total
            - exact_match_rate
            - avg_token_f1
            - results: List[EvalResult]
        """
        results: list[EvalResult] = []

        with torch.no_grad():
            for sample in samples:
                prompt = build_inference_prompt(sample, shuffle_facts=False)

                enc = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                    padding=True,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}

                # Greedy decoding for deterministic eval
                gen = self.model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                # Decode only newly generated tokens
                input_len = enc["input_ids"].shape[1]
                gen_tokens = gen[0, input_len:] if gen.shape[1] > input_len else gen[0]
                gen_answer = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

                gold = sample.answer.strip()
                norm_gen = normalize_text(gen_answer)
                norm_gold = normalize_text(gold)

                results.append(
                    EvalResult(
                        prompt=prompt,
                        generated=gen_answer,
                        gold=gold,
                        exact_match=(norm_gen == norm_gold),
                        token_f1=token_f1(norm_gen, norm_gold),
                    )
                )

        exact_count = sum(1 for r in results if r.exact_match)
        total = len(results)
        avg_f1 = float(np.mean([r.token_f1 for r in results])) if results else 0.0

        return {
            "exact_match_count": exact_count,
            "total": total,
            "exact_match_rate": exact_count / total if total else 0.0,
            "avg_token_f1": avg_f1,
            "results": results,
        }

    def save_metrics(self, metrics: dict, output_dir: str):
        """Save evaluation metrics to JSON."""
        os.makedirs(output_dir, exist_ok=True)

        # Save scalar metrics
        scalar = {k: v for k, v in metrics.items() if k != "results"}
        with open(os.path.join(output_dir, "eval_metrics.json"), "w") as f:
            json.dump(scalar, f, indent=2)

        # Save detailed results
        details = [
            {
                "prompt": r.prompt[:200],
                "generated": r.generated,
                "gold": r.gold,
                "exact_match": r.exact_match,
                "token_f1": r.token_f1,
            }
            for r in metrics["results"]
        ]
        with open(os.path.join(output_dir, "eval_detail.json"), "w") as f:
            json.dump(details, f, indent=2)

        print(f"Metrics saved to {output_dir}")
