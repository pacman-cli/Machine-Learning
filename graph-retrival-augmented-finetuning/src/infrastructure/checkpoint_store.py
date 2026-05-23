"""
Infrastructure: Local filesystem checkpoint storage.

Tracks best model by val_loss and manages checkpoint lifecycle.
"""

import json
import os

import torch

from src.domain.entities import TrainingMetrics
from src.domain.interfaces import CheckpointStore


class LocalCheckpointStore(CheckpointStore):
    """
    Saves/loads model checkpoints on local filesystem.

    Tracks best checkpoint via a metadata JSON file.
    """

    def __init__(self, base_dir: str = "./pt_graft_model_save"):
        self.base_dir = base_dir
        self._meta_path = os.path.join(base_dir, "checkpoint_meta.json")
        os.makedirs(base_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, epoch: int, metrics: TrainingMetrics) -> str:
        """Save model + optimizer state. Returns checkpoint path."""
        ckpt_dir = os.path.join(self.base_dir, f"checkpoint-epoch{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save model (HuggingFace format)
        model.save_pretrained(ckpt_dir)

        # Save optimizer state
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))

        # Update metadata
        meta = self._load_meta()
        meta["checkpoints"] = meta.get("checkpoints", [])
        meta["checkpoints"].append(
            {
                "epoch": epoch,
                "path": ckpt_dir,
                "train_loss": metrics.train_loss,
                "val_loss": metrics.val_loss,
                "perplexity": metrics.perplexity,
            }
        )

        # Track best
        if "best_val_loss" not in meta or metrics.val_loss < meta["best_val_loss"]:
            meta["best_val_loss"] = metrics.val_loss
            meta["best_path"] = ckpt_dir
            meta["best_epoch"] = epoch

        self._save_meta(meta)
        return ckpt_dir

    def load_best(self) -> str | None:
        """Return path to best checkpoint (lowest val_loss), or None."""
        meta = self._load_meta()
        return meta.get("best_path")

    def get_all_checkpoints(self) -> list:
        """Return list of all checkpoint metadata."""
        meta = self._load_meta()
        return meta.get("checkpoints", [])

    def _load_meta(self) -> dict:
        if os.path.exists(self._meta_path):
            with open(self._meta_path) as f:
                return json.load(f)
        return {}

    def _save_meta(self, meta: dict):
        with open(self._meta_path, "w") as f:
            json.dump(meta, f, indent=2)
