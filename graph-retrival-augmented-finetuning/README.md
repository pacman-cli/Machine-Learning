# GRAFT — Graph Retrieval-Augmented Fine-Tuning

Fine-tune causal language models to perform graph-grounded QA by learning to:
1. Parse scene graphs and knowledge base triples
2. Identify relevant reasoning paths through noisy context
3. Generate answers grounded in explicit graph traversals

## Architecture

```
┌─────────────────────────────────────────────────┐
│              scripts/ (Entry Points)             │
│   train.py, evaluate.py                         │
├─────────────────────────────────────────────────┤
│           src/application/ (Use Cases)           │
│   trainer.py, evaluator.py, inference.py        │
│   prompt_builder.py                             │
├─────────────────────────────────────────────────┤
│             src/domain/ (Core Logic)             │
│   entities.py, interfaces.py                    │
├─────────────────────────────────────────────────┤
│         src/infrastructure/ (I/O)               │
│   sample_loader.py, kb_retriever.py             │
└─────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install transformers torch datasets tqdm pyyaml

# Train
python -m scripts.train --config config/training_config.yaml

# Evaluate
python -m scripts.evaluate --model_path ./pt_graft_model_save/best
```

## Key Improvements Over Notebook

| Issue | Before | After |
|-------|--------|-------|
| Data diversity | 1 sample × 40 | 10 diverse samples × 4 |
| Label masking | Loss on full input | Loss only on answer tokens |
| LR scheduling | Flat rate | Linear warmup + decay |
| Architecture | Monolithic notebook | Clean layered modules |
| Retrieval | Static/hardcoded | Pluggable KBRetriever interface |
| Checkpointing | Every epoch | Best model tracking |
| Gradient clipping | None | max_norm=1.0 |

## Configuration

Edit `config/training_config.yaml` to adjust:
- Model selection
- Training hyperparameters
- Generation settings
- Output paths

## Project Structure

```
graph-retrival-augmented-finetuning/
├── config/
│   └── training_config.yaml
├── data/
│   └── sample_data.py          # Diverse training samples
├── src/
│   ├── domain/
│   │   ├── entities.py         # SceneGraph, KBTriple, GraftSample
│   │   └── interfaces.py      # SampleRepository, KBRetriever
│   ├── application/
│   │   ├── prompt_builder.py   # Graph → text serialization
│   │   ├── trainer.py          # Training loop with label masking
│   │   ├── evaluator.py        # Metrics: perplexity, EM, F1
│   │   └── inference.py        # Runtime prediction pipeline
│   └── infrastructure/
│       ├── sample_loader.py    # Raw dict → domain entities
│       └── kb_retriever.py     # Entity-matching fact retrieval
├── scripts/
│   ├── train.py
│   └── evaluate.py
└── Graph Retrieval-Augmented Fine-Tuning.ipynb  # Original notebook
```
