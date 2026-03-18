#!/usr/bin/env python3
"""
scripts/ingest.py
------------------
Standalone script to ingest documents into Endee.

Usage:
    python scripts/ingest.py --file data/knowledge_base.json
    python scripts/ingest.py --file mydata.txt --recreate
    python scripts/ingest.py --file docs.csv --index my_custom_index
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Endee vector DB")
    parser.add_argument("--file", required=True, help="Path to file (.json, .txt, .csv)")
    parser.add_argument("--index", default=None, help="Override index name")
    parser.add_argument("--recreate", action="store_true", help="Recreate index before ingesting")
    parser.add_argument("--host", default="http://localhost:8080", help="Endee host URL")
    args = parser.parse_args()

    kwargs = dict(
        endee_host=args.host,
        recreate_index=args.recreate,
    )
    if args.index:
        kwargs["index_name"] = args.index

    print(f"Connecting to Endee at {args.host}...")
    pipeline = RAGPipeline(**kwargs)

    print(f"Ingesting: {args.file}")
    count = pipeline.ingest_file(args.file)
    print(f"\nDone! {count} chunks stored in Endee.")

    stats = pipeline.stats()
    print(f"Index '{stats['index_name']}' now contains {stats['vector_count']} vectors.")


if __name__ == "__main__":
    main()
