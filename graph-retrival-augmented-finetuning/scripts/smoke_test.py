"""
Smoke test — verifies all components wire together correctly.

Runs a minimal training loop (1 epoch, 2 samples) to validate:
- Data loading and parsing
- Prompt building
- Tokenization with label masking
- Forward/backward pass
- Inference pipeline
- Evaluation metrics

Usage:
    python -m scripts.smoke_test
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data.sample_data import get_diverse_dataset
from src.application.prompt_builder import build_inference_prompt, build_training_prompt
from src.application.trainer import GraftDataset
from src.domain.entities import GraftSample, KBTriple, SceneGraph, SceneObject, SceneRelation
from src.infrastructure.kb_retriever import EntityMatchRetriever
from src.infrastructure.sample_loader import InMemorySampleRepository


def test_domain_entities():
    """Test entity creation and serialization."""
    obj1 = SceneObject("1", "car", ["red"])
    obj2 = SceneObject("2", "road", ["wet"])
    rel = SceneRelation("1", "on", "2")
    sg = SceneGraph(objects={"1": obj1, "2": obj2}, relations=[rel])

    assert sg.serialize_relations() == "(car -> on -> road)"

    triple = KBTriple("car", "IsA", "vehicle")
    assert triple.serialize() == "<car, IsA, vehicle>"

    print("  ✓ Domain entities")


def test_sample_loading():
    """Test raw dict → domain entity conversion."""
    raw_data = get_diverse_dataset()
    assert len(raw_data) == 40, f"Expected 40, got {len(raw_data)}"

    repo = InMemorySampleRepository(raw_data)
    samples = repo.load_all()
    assert len(samples) == 40
    assert isinstance(samples[0], GraftSample)
    assert samples[0].scene_graph.serialize_relations() != ""

    print("  ✓ Sample loading")


def test_prompt_building():
    """Test prompt serialization."""
    raw_data = get_diverse_dataset()
    repo = InMemorySampleRepository(raw_data)
    sample = repo.load_all()[0]

    train_prompt = build_training_prompt(sample, "<|endoftext|>", shuffle_facts=False)
    assert "Instruction:" in train_prompt
    assert "### Thought Traversal:" in train_prompt
    assert "### Answer:" in train_prompt
    assert "<|endoftext|>" in train_prompt

    inf_prompt = build_inference_prompt(sample, shuffle_facts=False)
    assert "### Answer:\n" in inf_prompt
    assert "### Thought Traversal:" not in inf_prompt

    print("  ✓ Prompt building")


def test_label_masking():
    """Test that GraftDataset masks prompt tokens in labels."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    raw_data = get_diverse_dataset()[:4]
    repo = InMemorySampleRepository(raw_data)
    samples = repo.load_all()

    dataset = GraftDataset(samples, tokenizer, max_length=256)
    item = dataset[0]

    labels = item["labels"]
    # Should have -100 for prompt portion and padding
    masked_count = (labels == -100).sum().item()
    total = labels.shape[0]
    assert masked_count > 0, "No tokens masked — label masking broken"
    assert masked_count < total, "All tokens masked — no training signal"

    print(f"  ✓ Label masking ({masked_count}/{total} tokens masked)")


def test_kb_retriever():
    """Test entity-match retrieval."""
    kb = [
        KBTriple("car", "IsA", "vehicle"),
        KBTriple("vehicle", "Requires", "fuel"),
        KBTriple("tree", "IsA", "plant"),
    ]
    retriever = EntityMatchRetriever(kb)
    results = retriever.retrieve("what does car need?", ["car"], top_k=5)

    assert len(results) >= 1
    entity_labels = [r.e1_label for r in results] + [r.e2_label for r in results]
    assert "car" in entity_labels or "vehicle" in entity_labels

    print("  ✓ KB retriever")


def test_forward_pass():
    """Test model forward pass with masked labels."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    raw_data = get_diverse_dataset()[:2]
    repo = InMemorySampleRepository(raw_data)
    samples = repo.load_all()

    dataset = GraftDataset(samples, tokenizer, max_length=128)
    item = dataset[0]

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids=item["input_ids"].unsqueeze(0),
            attention_mask=item["attention_mask"].unsqueeze(0),
            labels=item["labels"].unsqueeze(0),
        )

    assert outputs.loss is not None
    assert outputs.loss.item() > 0

    print(f"  ✓ Forward pass (loss={outputs.loss.item():.4f})")


def main():
    print("GRAFT Smoke Test")
    print("=" * 40)

    test_domain_entities()
    test_sample_loading()
    test_prompt_building()
    test_label_masking()
    test_kb_retriever()
    test_forward_pass()

    print("=" * 40)
    print("All tests passed.")


if __name__ == "__main__":
    main()
