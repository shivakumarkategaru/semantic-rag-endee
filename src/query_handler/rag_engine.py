"""
query_handler/rag_engine.py
----------------------------
Retrieval-Augmented Generation (RAG) engine.

Full pipeline:
    query text
      → NLP preprocessing (clean + normalize)
      → encode query  (EmbeddingEncoder)
      → similarity search  (VectorStore / Endee)
      → retrieve top-k context chunks
      → generate answer  (local LLM via transformers, or template fallback)
      → RAGResponse

The LLM step uses a lightweight generative model (GPT-2 or facebook/opt-125m)
so the system runs entirely offline on CPU. Users with a GPU can swap in any
HuggingFace causal-LM by changing GENERATOR_MODEL in config.py.
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..data_processing.processor import TextCleaner
from ..embeddings.encoder import EmbeddingEncoder
from ..database.vector_store import VectorStore, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """
    Result returned by RAGEngine.query().

    Attributes:
        question:       The original user question.
        answer:         Generated or extracted answer text.
        sources:        Top search results used as context.
        context_texts:  Plain text chunks fed to the generator.
        model_used:     Which generator model produced the answer.
    """
    question: str
    answer: str
    sources: List[SearchResult]
    context_texts: List[str] = field(default_factory=list)
    model_used: str = "template"

    def display(self) -> str:
        """Format the response for human-readable terminal output."""
        divider = "─" * 70
        lines = [
            divider,
            f"  QUESTION : {self.question}",
            divider,
            f"  ANSWER   : {textwrap.fill(self.answer, width=70, subsequent_indent='             ')}",
            divider,
            f"  SOURCES  ({len(self.sources)} chunks retrieved):",
        ]
        for i, src in enumerate(self.sources, 1):
            score_bar = "█" * int(src.score * 10) + "░" * (10 - int(src.score * 10))
            lines.append(
                f"    [{i}] Score {src.score:.3f} {score_bar}  "
                f"[{src.source} / chunk {src.chunk_index}]"
            )
            lines.append(
                "        " + textwrap.shorten(src.text, width=80, placeholder="...")
            )
        lines.append(divider)
        return "\n".join(lines)


class GeneratorMixin:
    """
    Optional LLM-based answer generation using HuggingFace transformers.
    Falls back to a deterministic template if the library is unavailable.
    """

    _pipeline = None
    _generator_model: str = ""

    def _try_load_generator(self, model_name: str) -> bool:
        """
        Attempt to load a causal-LM text-generation pipeline.
        Returns True if successful.
        """
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading generator model: %s", model_name)
            self._pipeline = hf_pipeline(
                "text-generation",
                model=model_name,
                device=-1,          # CPU
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=50256,
            )
            self._generator_model = model_name
            logger.info("Generator model loaded: %s", model_name)
            return True
        except Exception as exc:
            logger.warning("Could not load generator model '%s': %s", model_name, exc)
            return False

    def _generate_with_llm(self, prompt: str) -> str:
        """Run the loaded pipeline and extract the generated continuation."""
        outputs = self._pipeline(prompt, return_full_text=False)
        raw = outputs[0]["generated_text"].strip()
        # Return full generated text to ensure complete answers
        return raw

    @staticmethod
    def _template_answer(question: str, context_texts: List[str]) -> str:
        """
        Deterministic fallback: extract the most relevant sentence from context
        using keyword overlap (no LLM required).
        """
        if not context_texts:
            return "No relevant information found."

        query_words = set(question.lower().split())
        best_sentence = ""
        best_overlap = -1

        for chunk in context_texts:
            for sentence in chunk.split("."):
                sentence = sentence.strip()
                if not sentence:
                    continue
                # Fixing type checker lint by using .intersection()
                overlap = len(query_words.intersection(sentence.lower().split()))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sentence = sentence

        if best_sentence:
            return best_sentence.strip() + "."
        return context_texts[0][:300].strip()


class RAGEngine(GeneratorMixin):
    """
    Orchestrates the full Retrieval-Augmented Generation pipeline.

    Usage:
        engine = RAGEngine(encoder, vector_store)
        engine.load_generator()          # optional, adds LLM step
        response = engine.query("What is the capital of France?")
        print(response.display())
    """

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        vector_store: VectorStore,
        top_k: int = 3,
        ef: int = 128,
        min_score: float = 0.5,
    ):
        self.encoder = encoder
        self.vector_store = vector_store
        self.top_k = top_k
        self.ef = ef
        self.min_score = min_score
        self._cleaner = TextCleaner()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_generator(self, model_name: str = "gpt2") -> bool:
        """
        Optionally attach an LLM for answer synthesis.

        Recommended lightweight models (all Apache/MIT licensed):
          - "gpt2"                 (~500 MB, widely available)
          - "facebook/opt-125m"    (~250 MB, faster)
          - "distilgpt2"           (~350 MB, smallest)

        Returns True if the model loaded successfully.
        """
        return self._try_load_generator(model_name)

    # ------------------------------------------------------------------
    # Query pipeline
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filters: Optional[List] = None,
        min_score: Optional[float] = None,
    ) -> RAGResponse:
        """
        Execute the full RAG pipeline for a user question.

        Steps:
            1. NLP preprocess the question
            2. Encode question → dense vector
            3. Search Endee for top-k similar chunks
            4. Assemble context
            5. Generate / extract answer
        """
        top_k = top_k or self.top_k
        min_score = min_score if min_score is not None else self.min_score

        # Step 1 – preprocess
        clean_q = self._cleaner.clean(question)

        # Step 2 – encode
        logger.debug("Encoding query: %s", clean_q)
        query_vector = self.encoder.encode(clean_q)

        # Step 3 – retrieve
        logger.debug("Searching Endee (top_k=%d)...", top_k)
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            ef=self.ef,
            filters=filters,
        )

        # Filter by threshold and limit top chunks
        results = [r for r in results if r.score >= min_score]
        results = results[:3]

        # Step 4 – build context
        context_texts = [r.text for r in results]

        # Step 5 – answer
        if hasattr(self, "_openai_gen") and self._openai_gen:
            answer = self._openai_gen.generate(question, context_texts)
            model_used = "groq-llama"
        else:
            answer, model_used = self._synthesize(question, context_texts)

        return RAGResponse(
            question=question,
            answer=answer,
            sources=results,
            context_texts=context_texts,
            model_used=model_used,
        )

    def semantic_search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Pure semantic search — returns ranked chunks without answer synthesis.
        Useful for document retrieval / recommendation use cases.
        """
        top_k = top_k or self.top_k
        clean_q = self._cleaner.clean(query)
        query_vector = self.encoder.encode(clean_q)
        return self.vector_store.search(query_vector=query_vector, top_k=top_k)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, question: str, context_texts: List[str]) -> str:
        """Assemble a prompt from question + retrieved context."""
        if not context_texts:
            return (
                "You are a professional AI assistant. Answer clearly and accurately.\n\n"
                f"Question: {question}\n"
                "Answer:"
            )

        context_block = "\n\n".join(
            f"[Chunk {i+1}]\n{text.strip()}" for i, text in enumerate(context_texts[:3])
        )
        return (
            "You are a professional AI assistant. Answer clearly and accurately using the provided context. "
            "Do not copy text directly. Explain in your own words. If the context is insufficient, say that "
            "you do not have enough information.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def _synthesize(self, question: str, context_texts: List[str]) -> Tuple[str, str]:
        """
        Return (answer_text, model_name).
        Uses LLM if available, falls back to template extraction.
        """
        if self._pipeline is not None:
            try:
                prompt = self._build_prompt(question, context_texts)
                answer = self._generate_with_llm(prompt)
                return answer, self._generator_model
            except Exception as exc:
                logger.warning("LLM generation failed: %s. Falling back to template.", exc)

        answer = self._template_answer(question, context_texts)
        return answer, "template"
