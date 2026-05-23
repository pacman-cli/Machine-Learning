"""
Infrastructure: Load GRAFT samples from raw data format.

Converts raw dict-based dataset into domain entities.
"""

from src.domain.entities import (
    GraftSample,
    KBTriple,
    SceneGraph,
    SceneObject,
    SceneRelation,
)
from src.domain.interfaces import SampleRepository


def parse_raw_sample(raw: dict) -> GraftSample:
    """Convert a raw dict sample into a GraftSample entity."""
    # Parse scene graph
    sg_data = raw["gqa_scene_graph"]
    objects = {
        obj_id: SceneObject(
            object_id=obj_id,
            name=obj_data["name"],
            attributes=obj_data.get("attributes", []),
        )
        for obj_id, obj_data in sg_data["objects"].items()
    }
    relations = [
        SceneRelation(
            source_id=rel["source"],
            relation_name=rel["name"],
            target_id=rel["target"],
        )
        for rel in sg_data["relations"]
    ]
    scene_graph = SceneGraph(objects=objects, relations=relations)

    # Parse KB facts
    oracle_facts = [
        KBTriple(e1_label=f["e1_label"], relation=f["rel"], e2_label=f["e2_label"])
        for f in raw.get("fvqa_graph_rag_facts", [])
    ]
    distractor_facts = [
        KBTriple(e1_label=f["e1_label"], relation=f["rel"], e2_label=f["e2_label"])
        for f in raw.get("kb_distractor_pool", [])
    ]

    return GraftSample(
        image_id=raw.get("image_id", "unknown"),
        scene_graph=scene_graph,
        oracle_facts=oracle_facts,
        distractor_facts=distractor_facts,
        query=raw.get("query", ""),
        ground_truth_trajectory=raw.get("ground_truth_trajectory", ""),
        answer=raw.get("answer", ""),
    )


class InMemorySampleRepository(SampleRepository):
    """Simple in-memory sample store. Good for notebook/prototyping."""

    def __init__(self, raw_data: list[dict]):
        self._samples = [parse_raw_sample(r) for r in raw_data]

    def load_all(self) -> list[GraftSample]:
        return self._samples

    def save(self, samples: list[GraftSample]) -> None:
        self._samples = samples
