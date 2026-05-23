"""
Domain entities for GRAFT (Graph Retrieval-Augmented Fine-Tuning).

Pure data classes — no external dependencies, no I/O.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SceneObject:
    """A single object in a GQA scene graph."""

    object_id: str
    name: str
    attributes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SceneRelation:
    """A directed edge between two scene objects."""

    source_id: str
    relation_name: str
    target_id: str


@dataclass
class SceneGraph:
    """GQA-style scene graph with objects and relations."""

    objects: dict[str, SceneObject] = field(default_factory=dict)
    relations: list[SceneRelation] = field(default_factory=list)

    def serialize_relations(self) -> str:
        """Convert relations to text: (src -> rel -> tgt), ..."""
        parts = []
        for rel in self.relations:
            src_name = self.objects[rel.source_id].name
            tgt_name = self.objects[rel.target_id].name
            parts.append(f"({src_name} -> {rel.relation_name} -> {tgt_name})")
        return ", ".join(parts)


@dataclass(frozen=True)
class KBTriple:
    """A knowledge base triple (entity1, relation, entity2)."""

    e1_label: str
    relation: str
    e2_label: str

    def serialize(self) -> str:
        return f"<{self.e1_label}, {self.relation}, {self.e2_label}>"


@dataclass
class GraftSample:
    """A single GRAFT training/inference sample."""

    image_id: str
    scene_graph: SceneGraph
    oracle_facts: list[KBTriple] = field(default_factory=list)
    distractor_facts: list[KBTriple] = field(default_factory=list)
    query: str = ""
    ground_truth_trajectory: str = ""
    answer: str = ""

    @property
    def all_facts(self) -> list[KBTriple]:
        """Oracle + distractor facts combined."""
        return self.oracle_facts + self.distractor_facts


@dataclass
class TrainingMetrics:
    """Metrics from a training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    perplexity: float | None = None


@dataclass
class EvalResult:
    """Result from evaluating a single sample."""

    prompt: str
    generated: str
    gold: str
    exact_match: bool
    token_f1: float
