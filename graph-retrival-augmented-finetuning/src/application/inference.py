"""
Inference pipeline — query → answer using trained GRAFT model.

This is the runtime component that:
1. Takes a scene graph + query
2. Retrieves relevant KB facts (via KBRetriever)
3. Builds prompt
4. Generates answer with thought traversal
"""

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.application.prompt_builder import build_inference_prompt
from src.domain.entities import GraftSample, KBTriple, SceneGraph
from src.domain.interfaces import KBRetriever


@dataclass
class InferenceResult:
    """Result from a single inference call."""

    query: str
    generated_text: str
    thought_traversal: str
    answer: str


class GraftInferencePipeline:
    """End-to-end inference: scene graph + query → grounded answer."""

    def __init__(
        self,
        model_path: str,
        base_model_id: str = "distilgpt2",
        kb_retriever: KBRetriever | None = None,
        max_new_tokens: int = 64,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.95,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

        # Try loading tokenizer from checkpoint; fall back to base model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.kb_retriever = kb_retriever
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p

    def predict(
        self,
        scene_graph: SceneGraph,
        query: str,
        known_facts: list[KBTriple] | None = None,
    ) -> InferenceResult:
        """
        Generate answer for a query given a scene graph.

        If kb_retriever is set and known_facts is None, retrieves facts dynamically.
        """
        # Get facts
        if known_facts is not None:
            facts = known_facts
        elif self.kb_retriever is not None:
            entities = [obj.name for obj in scene_graph.objects.values()]
            facts = self.kb_retriever.retrieve(query, entities)
        else:
            facts = []

        # Build sample for prompt construction
        sample = GraftSample(
            image_id="inference",
            scene_graph=scene_graph,
            oracle_facts=facts,
            distractor_facts=[],
            query=query,
        )

        prompt = build_inference_prompt(sample, shuffle_facts=False)

        # Generate
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            gen = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                top_k=self.top_k,
                top_p=self.top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only new tokens
        input_len = enc["input_ids"].shape[1]
        gen_tokens = gen[0, input_len:] if gen.shape[1] > input_len else gen[0]
        generated = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Parse thought traversal vs answer
        thought = ""
        answer = generated
        if "### Answer:" in generated:
            parts = generated.split("### Answer:")
            thought = parts[0].strip()
            answer = parts[1].strip() if len(parts) > 1 else ""

        return InferenceResult(
            query=query,
            generated_text=generated,
            thought_traversal=thought,
            answer=answer,
        )
