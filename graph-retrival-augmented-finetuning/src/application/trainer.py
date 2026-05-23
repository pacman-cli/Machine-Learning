"""
Training orchestration — handles the full training loop with:
- Train/val split
- Label masking (only compute loss on answer tokens, not prompt)
- Learning rate scheduling
- Early stopping
- Checkpointing best model
"""

import math
import os

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.application.prompt_builder import build_training_prompt
from src.domain.entities import GraftSample, TrainingMetrics


class GraftDataset(Dataset):
    """Tokenized GRAFT dataset with proper label masking."""

    def __init__(
        self,
        samples: list[GraftSample],
        tokenizer,
        max_length: int = 512,
        shuffle_facts: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_facts = shuffle_facts
        self.samples = samples

        # Pre-tokenize all samples
        self.encodings = self._tokenize_all()

    def _tokenize_all(self):
        """Tokenize and create label-masked tensors."""
        texts = [
            build_training_prompt(s, self.tokenizer.eos_token, self.shuffle_facts)
            for s in self.samples
        ]

        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Create labels with prompt masking
        # Find where "### Thought Traversal:\n" starts — only train on tokens after that
        labels = encodings["input_ids"].clone()
        answer_marker = "### Thought Traversal:\n"

        for i, text in enumerate(texts):
            # Find the character position of the answer section
            marker_pos = text.find(answer_marker)
            if marker_pos == -1:
                continue

            # Tokenize just the prompt portion to find token boundary
            prompt_portion = text[: marker_pos + len(answer_marker)]
            prompt_tokens = self.tokenizer(
                prompt_portion,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            prompt_len = prompt_tokens["input_ids"].shape[1]

            # Mask prompt tokens in labels (set to -100 so loss ignores them)
            # Account for left-padding: find where real tokens start
            attention_mask = encodings["attention_mask"][i]
            pad_len = (attention_mask == 0).sum().item()
            mask_end = pad_len + prompt_len
            labels[i, :mask_end] = -100

        # Also mask padding tokens
        labels[encodings["attention_mask"] == 0] = -100

        encodings["labels"] = labels
        return encodings

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.encodings["labels"][idx],
        }


class GraftTrainer:
    """Orchestrates GRAFT model training with best practices."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        model_name = config["model"]["name"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        # Output directory
        self.output_dir = config["output"]["dir"]
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self, samples: list[GraftSample]) -> list[TrainingMetrics]:
        """
        Full training loop.

        Returns list of per-epoch metrics.
        """
        cfg = self.config["training"]

        # Set seed
        torch.manual_seed(cfg["seed"])

        # Build dataset
        dataset = GraftDataset(
            samples=samples,
            tokenizer=self.tokenizer,
            max_length=self.config["model"]["max_length"],
            shuffle_facts=self.config["retrieval"]["shuffle_facts"],
        )

        # Train/val split
        total = len(dataset)
        train_len = int(total * cfg["train_split"])
        val_len = total - train_len
        train_ds, val_ds = random_split(dataset, [train_len, val_len])

        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

        # Optimizer + scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )

        total_steps = len(train_loader) * cfg["epochs"]
        warmup_steps = int(total_steps * cfg["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training loop
        best_val_loss = float("inf")
        all_metrics: list[TrainingMetrics] = []

        for epoch in range(1, cfg["epochs"] + 1):
            # Train
            train_loss = self._train_epoch(train_loader, optimizer, scheduler, epoch)

            # Validate
            val_loss = self._validate(val_loader, epoch)

            perplexity = math.exp(val_loss) if val_loss < 100 else float("inf")
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                perplexity=perplexity,
            )
            all_metrics.append(metrics)

            print(
                f"Epoch {epoch} — train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, perplexity={perplexity:.4f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, "best")
                print(f"  → New best model saved (val_loss={val_loss:.4f})")

            # Always save latest checkpoint
            if not self.config["output"]["save_best_only"]:
                self._save_checkpoint(epoch, f"checkpoint-epoch{epoch}")

        # Save final
        self._save_checkpoint(cfg["epochs"], "final")
        print(f"Training complete. Final model: {self.output_dir}/final")

        return all_metrics

    def _train_epoch(self, loader: DataLoader, optimizer, scheduler, epoch: int) -> float:
        """Single training epoch. Returns average loss."""
        if len(loader) == 0:
            return 0.0
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch} [train]")
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],  # Masked labels — loss only on answer tokens
            )

            loss = outputs.loss
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(loader)

    def _validate(self, loader: DataLoader, epoch: int) -> float:
        """Validation pass. Returns average loss."""
        if len(loader) == 0:
            return 0.0
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Epoch {epoch} [val]"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                total_loss += outputs.loss.item()

        return total_loss / len(loader)

    def _save_checkpoint(self, epoch: int, name: str):
        """Save model checkpoint."""
        ckpt_dir = os.path.join(self.output_dir, name)
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)
