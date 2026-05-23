"""
Train GRAFT model from config.

Usage:
    python -m scripts.train --config config/training_config.yaml
"""

import argparse
import os
import sys

import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.sample_data import get_diverse_dataset
from src.application.trainer import GraftTrainer
from src.infrastructure.sample_loader import InMemorySampleRepository


def main():
    parser = argparse.ArgumentParser(description="Train GRAFT model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training config YAML",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load data
    raw_data = get_diverse_dataset()
    repo = InMemorySampleRepository(raw_data)
    samples = repo.load_all()

    print(f"Loaded {len(samples)} samples")
    print(f"Model: {config['model']['name']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print()

    # Train
    trainer = GraftTrainer(config)
    metrics = trainer.train(samples)

    # Summary
    print("\n=== Training Summary ===")
    for m in metrics:
        print(
            f"  Epoch {m.epoch}: train={m.train_loss:.4f}, val={m.val_loss:.4f}, ppl={m.perplexity:.4f}"
        )


if __name__ == "__main__":
    main()
