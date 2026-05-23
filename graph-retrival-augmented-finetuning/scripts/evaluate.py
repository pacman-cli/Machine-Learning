"""
Evaluate trained GRAFT model.

Usage:
    python -m scripts.evaluate --model_path ./pt_graft_model_save/best
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.sample_data import get_diverse_dataset
from src.application.evaluator import GraftEvaluator
from src.infrastructure.sample_loader import InMemorySampleRepository


def main():
    parser = argparse.ArgumentParser(description="Evaluate GRAFT model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./pt_graft_model_save/best",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="distilgpt2",
        help="Base model ID for tokenizer",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Max samples to evaluate",
    )
    args = parser.parse_args()

    # Load validation data (last 20% of dataset)
    raw_data = get_diverse_dataset()
    split_idx = int(len(raw_data) * 0.8)
    val_data = raw_data[split_idx:][: args.max_samples]

    repo = InMemorySampleRepository(val_data)
    samples = repo.load_all()

    print(f"Evaluating {len(samples)} samples from: {args.model_path}")

    # Evaluate
    evaluator = GraftEvaluator(args.model_path, args.base_model)
    metrics = evaluator.evaluate(samples)

    # Print results
    print("\n=== Evaluation Results ===")
    print(
        f"  Exact Match: {metrics['exact_match_count']}/{metrics['total']} "
        f"({metrics['exact_match_rate']:.2%})"
    )
    print(f"  Avg Token-F1: {metrics['avg_token_f1']:.4f}")

    # Show failures
    failures = [r for r in metrics["results"] if not r.exact_match]
    if failures:
        print("\n  Showing up to 5 failures:")
        for r in failures[:5]:
            print(f"    Gold: {r.gold[:80]}")
            print(f"    Gen:  {r.generated[:80]}")
            print(f"    F1:   {r.token_f1:.4f}")
            print()

    # Save
    evaluator.save_metrics(metrics, args.model_path)


if __name__ == "__main__":
    main()
