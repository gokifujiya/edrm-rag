from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description = "Display chunks from a processed corpus"
    )
    parser.add_argument(
        "--corpus",
        default = "edrm",
        help = "Corpus under data/processed",
    )
    parser.add_argument(
        "--start",
        type = int,
        default = 0,
        help = "First chunk to display",
    )
    parser.add_argument(
        "--count",
        type = int,
        default = 5,
        help = "Number of chunks to display",
    )

    args = parser.parse_args()

    path = (
        ROOT
        / "data"
        / "processed"
        / args.corpus
        / "chunks.jsonl"
    )

    if not path.exists():
        raise SystemExit(f"Missing: {path}")

    chunks = []

    with path.open("r", encoding = "utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    end = min(args.start + args.count, len(chunks))

    print(f"\nCorpus: {args.corpus}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Showing: {args.start} to {end - 1}\n")

    for i in range(args.start, end):
        chunk = chunks[i]

        print("=" * 80)
        print(f"CHUNK #{i}")
        print(f"Chunk ID:    {chunk.get('chunk_id')}")
        print(f"Source:      {chunk.get('source_path')}")
        print(f"File type:   {chunk.get('file_type')}")
        print(f"Title:       {chunk.get('title')}")
        print(f"Chunk index: {chunk.get('chunk_index')}")
        print("-" * 80)
        print(chunk.get("text", ""))
        print()


if __name__ == "__main__":
    main()
