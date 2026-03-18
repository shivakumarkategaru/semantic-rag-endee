#!/usr/bin/env python3
"""
scripts/query.py
-----------------
One-shot query script for the Semantic RAG system.

Usage:
    python scripts/query.py "What is machine learning?"
    python scripts/query.py "Explain transformers" --top-k 3
    python scripts/query.py "Python language" --search-only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument("question", help="Question or search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--search-only", action="store_true", help="Return chunks, skip answer synthesis")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM answer generation")
    parser.add_argument("--host", default="http://localhost:8080", help="Endee host URL")
    args = parser.parse_args()

    pipeline = RAGPipeline(endee_host=args.host, use_llm=args.use_llm)

    if args.search_only:
        results = pipeline.search(args.question, top_k=args.top_k)
        print(f"\nTop {len(results)} results for: '{args.question}'\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] Score: {r.score:.4f} | Source: {r.source} | Chunk: {r.chunk_index}")
            print(f"    {r.text[:200]}...")
            print()
    else:
        response = pipeline.ask(args.question, top_k=args.top_k)
        print(response.display())


if __name__ == "__main__":
    main()
