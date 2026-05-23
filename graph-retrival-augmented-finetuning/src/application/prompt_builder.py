"""
Prompt composition — serializes graph structures into GRAFT training prompts.

Handles:
- Scene graph → text
- KB facts (oracle + distractors) → shuffled text
- Full prompt assembly with instruction template
"""

import random

from src.domain.entities import GraftSample, KBTriple

INSTRUCTION = (
    "Isolate the correct reasoning path through the noisy graph context to answer the query."
)


def serialize_facts(facts: list[KBTriple], shuffle: bool = True) -> str:
    """Serialize KB triples to semicolon-separated string, optionally shuffled."""
    serialized = [f.serialize() for f in facts]
    if shuffle:
        random.shuffle(serialized)
    return "; ".join(serialized)


def build_training_prompt(sample: GraftSample, eos_token: str, shuffle_facts: bool = True) -> str:
    """
    Build full training prompt WITH ground truth (for fine-tuning).

    Format:
        Instruction: ...
        Scene Graph: (src -> rel -> tgt)
        Graph-RAG Facts: <e1, rel, e2>; ...
        Query: ...
        ### Thought Traversal:
        <trajectory>
        ### Answer:
        <answer><eos>
    """
    sg_context = sample.scene_graph.serialize_relations()
    facts_context = serialize_facts(sample.all_facts, shuffle=shuffle_facts)

    return (
        f"Instruction: {INSTRUCTION}\n"
        f"Scene Graph: {sg_context}\n"
        f"Graph-RAG Facts: {facts_context}\n"
        f"Query: {sample.query}\n"
        f"### Thought Traversal:\n{sample.ground_truth_trajectory}\n"
        f"### Answer:\n{sample.answer}{eos_token}"
    )


def build_inference_prompt(sample: GraftSample, shuffle_facts: bool = False) -> str:
    """
    Build inference prompt WITHOUT ground truth (for generation).

    Stops at "### Answer:\n" — model generates the rest.
    """
    sg_context = sample.scene_graph.serialize_relations()
    facts_context = serialize_facts(sample.all_facts, shuffle=shuffle_facts)

    return (
        f"Instruction: {INSTRUCTION}\n"
        f"Scene Graph: {sg_context}\n"
        f"Graph-RAG Facts: {facts_context}\n"
        f"Query: {sample.query}\n"
        f"### Answer:\n"
    )
