"""
Infrastructure: Knowledge Base retrieval.

Provides actual retrieval of KB facts at inference time.
Current implementation: simple entity-matching retriever.
Future: FAISS embedding-based retrieval.
"""

from src.domain.entities import KBTriple
from src.domain.interfaces import KBRetriever


class EntityMatchRetriever(KBRetriever):
    """
    Simple retriever: returns KB triples where entity labels
    overlap with scene graph entities.

    Good baseline. Replace with embedding-based retrieval for production.
    """

    def __init__(self, knowledge_base: list[KBTriple]):
        self._kb = knowledge_base
        # Build index: entity_name → list of triples containing it
        self._index = {}
        for triple in self._kb:
            for entity in [triple.e1_label, triple.e2_label]:
                if entity not in self._index:
                    self._index[entity] = []
                self._index[entity].append(triple)

    def retrieve(self, query: str, scene_entities: list[str], top_k: int = 5) -> list[KBTriple]:
        """Retrieve triples matching scene entities."""
        scored = {}
        for entity in scene_entities:
            # Exact match
            for triple in self._index.get(entity, []):
                key = (triple.e1_label, triple.relation, triple.e2_label)
                scored[key] = scored.get(key, 0) + 1

            # Partial match (entity name contained in triple entity)
            for kb_entity, triples in self._index.items():
                if entity in kb_entity or kb_entity in entity:
                    for triple in triples:
                        key = (triple.e1_label, triple.relation, triple.e2_label)
                        scored[key] = scored.get(key, 0) + 0.5

        # Sort by score, return top_k
        sorted_keys = sorted(scored.keys(), key=lambda k: scored[k], reverse=True)[:top_k]
        return [KBTriple(e1_label=k[0], relation=k[1], e2_label=k[2]) for k in sorted_keys]
