#!/usr/bin/env python3
"""
main.py
--------
Entry point for the Semantic RAG system with Endee vector database.

Demonstrates the full pipeline:
  1. Connect to Endee (running via Docker)
  2. Ingest the sample knowledge base
  3. Run several example queries
  4. Interactive Q&A loop

Run:
    python main.py                  # interactive mode after demo
    python main.py --ingest-only    # ingest data and exit
    python main.py --no-interactive # run demo queries and exit
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so src imports work when run directly
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


DEMO_QUESTIONS = [
    "What is Retrieval-Augmented Generation?",
    "How do vector databases work?",
    "What is the Transformer architecture in deep learning?",
    "Explain the difference between machine learning and deep learning.",
    "What are embeddings and why are they useful?",
]


def run_demo(pipeline: RAGPipeline) -> None:
    """Execute pre-defined demo questions and print results."""
    print("\n" + "=" * 70)
    print("  DEMO: Semantic RAG with Endee Vector Database")
    print("=" * 70)

    stats = pipeline.stats()
    print(f"\n  Index     : {stats['index_name']}")
    print(f"  Vectors   : {stats['vector_count']}")
    print(f"  Dimension : {stats['dimension']}")
    print(f"  Model     : {stats['embedding_model']}")
    print()

    for q in DEMO_QUESTIONS:
        response = pipeline.ask(q)
        print(response.display())
        print()


def interactive_loop(pipeline: RAGPipeline) -> None:
    """Simple REPL for user questions."""
    print("\n" + "=" * 70)
    print("  Interactive Mode  (type 'quit' or 'exit' to stop)")
    print("  Commands: 'search <query>' for raw retrieval, or ask any question")
    print("=" * 70 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        lower = user_input.lower()
        if lower in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        if lower.startswith("search "):
            query = user_input[7:].strip()
            results = pipeline.search(query, top_k=3)
            print(f"\n  Top {len(results)} results for: '{query}'")
            for i, r in enumerate(results, 1):
                print(f"  [{i}] {r.score:.3f}  [{r.source}]  {r.text[:120]}...")
            print()
        else:
            response = pipeline.ask(user_input)
            print(response.display())
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Semantic RAG with Endee Vector Database"
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Ingest data and exit without running queries.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Run demo queries then exit without entering interactive mode.",
    )
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Drop existing Endee index and recreate it from scratch.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Load a local LLM for answer generation (requires transformers).",
    )
    parser.add_argument(
        "--data",
        default="data/knowledge_base.json",
        help="Path to the knowledge base file (.json, .txt, or .csv).",
    )
    args = parser.parse_args()

    # ── Initialise pipeline ──────────────────────────────────────────
    print("Starting Semantic RAG Pipeline...")
    print(f"  Knowledge base: {args.data}")
    print(f"  Use LLM: {args.use_llm}")
    print()

    pipeline = RAGPipeline(
        recreate_index=args.recreate_index,
        use_llm=args.use_llm,
    )

    # ── Ingest knowledge base ────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    count = pipeline.ingest_file(str(data_path))
    print(f"\n  Ingested {count} document chunks into Endee.\n")

    if args.ingest_only:
        print("Ingest complete. Exiting.")
        return

    # ── Demo queries ─────────────────────────────────────────────────
    run_demo(pipeline)

    # ── Interactive mode ─────────────────────────────────────────────
    if not args.no_interactive:
        interactive_loop(pipeline)


if __name__ == "__main__":
    main()
