"""
Domain interfaces — implemented by infrastructure layer.

These define WHAT the system needs, not HOW it's done.
"""

from abc import ABC, abstractmethod

from src.domain.entities import GraftSample, KBTriple, TrainingMetrics


class SampleRepository(ABC):
    """Load and store GRAFT samples."""

    @abstractmethod
    def load_all(self) -> list[GraftSample]:
        """Load all available samples."""
        ...

    @abstractmethod
    def save(self, samples: list[GraftSample]) -> None:
        """Persist samples."""
        ...


class KBRetriever(ABC):
    """Retrieve knowledge base facts given a query context."""

    @abstractmethod
    def retrieve(self, query: str, scene_entities: list[str], top_k: int = 5) -> list[KBTriple]:
        """
        Retrieve relevant KB triples for a query.

        Args:
            query: The natural language question.
            scene_entities: Entity names from the scene graph.
            top_k: Max number of facts to retrieve.

        Returns:
            List of relevant KBTriple objects.
        """
        ...


class CheckpointStore(ABC):
    """Save and load model checkpoints."""

    @abstractmethod
    def save_checkpoint(self, model, optimizer, epoch: int, metrics: TrainingMetrics) -> str:
        """Save checkpoint, return path."""
        ...

    @abstractmethod
    def load_best(self) -> str | None:
        """Load path to best checkpoint (lowest val_loss)."""
        ...
