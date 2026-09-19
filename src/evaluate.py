"""Evaluate retrieval quality for a Chroma corpus.

Examples:
    python src/evaluate.py
    python src/evaluate.py --corpus edrm
    python src/evaluate.py --corpus edrm -k 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder


ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index"
EVAL_DIR = ROOT / "eval"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
CANDIDATE_K = 10


def load_evaluation_set(corpus: str) -> list[dict]:
    path = EVAL_DIR / f"{corpus}_queries.jsonl"

    if not path.exists():
        raise SystemExit(f"Evaluation file does not exist: {path}")

    items = []

    with path.open("r", encoding = "utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    if not items:
        raise SystemExit(f"Evaluation file is empty: {path}")

    return items


def get_collection(corpus: str):
    client = chromadb.PersistentClient(path = str(INDEX_DIR))

    try:
        return client.get_collection(corpus)
    except Exception:
        raise SystemExit(
            f"Chroma collection '{corpus}' does not exist.\n"
            f"Build it first with:\n"
            f"python src/index.py build --corpus {corpus}"
        )


def evaluate(corpus: str, k: int = 5) -> None:
    model = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL)
    collection = get_collection(corpus)

    if collection.count() == 0:
        raise SystemExit(f"collection '{corpus}' is empty")

    evaluation_set = load_evaluation_set(corpus)

    results = []

    for item in evaluation_set:
        query_id = item["id"]
        query = item["query"]
        relevant_ids = set(item["relevant_chunk_ids"])

        vector = model.encode(
            [query],
            normalize_embeddings = True,
        )[0].tolist()

        # Step 1: retrieve 10 candidates with vector search
        candidate_count = min(CANDIDATE_K, collection.count())

        result = collection.query(
            query_embeddings=[vector],
            n_results=candidate_count,
            include=["documents", "metadatas", "distances"],
        )

        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        ids = result["ids"][0]

        # Step 2: rerank the candidates
        pairs = [
            [query, document]
            for document in documents
        ]

        rerank_scores = reranker.predict(pairs)

        candidates = []

        for chunk_id, metadata, distance, rerank_score in zip(
            ids,
            metadatas,
            distances,
            rerank_scores,
        ):
            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "title": metadata.get("title", ""),
                    "source_path": metadata.get("source_path", ""),
                    "vector_score": 1.0 - float(distance),
                    "rerank_score": float(rerank_score),
                }
            )

        # Step 3: sort by reranker score
        candidates.sort(
            key=lambda hit: hit["rerank_score"],
            reverse=True,
        )

        # Step 4: keep final top-k
        retrieved = []

        for rank, hit in enumerate(candidates[:k], start=1):
            hit["rank"] = rank
            retrieved.append(hit)

        hit_rank = None

        for hit in retrieved:
            if hit["chunk_id"] in relevant_ids:
                hit_rank = hit["rank"]
                break

        results.append(
            {
                "id": query_id,
                "query": query,
                "relevant_chunk_ids": relevant_ids,
                "hit_rank": hit_rank,
                "retrieved": retrieved,
            }
        )

    print(f"\nCorpus: {corpus}")
    print(f"Queries: {len(results)}")
    print(f"Top-k: {k}\n")

    for number, result in enumerate(results, start=1):
        print("=" * 70)
        print(f"[{number}] {result['id']}: {result['query']}")
        print(f"Relevant chunks: {len(result['relevant_chunk_ids'])}")

        if result["hit_rank"] is not None:
            print(f"Result: HIT at rank {result['hit_rank']}")
        else:
            print("Result: MISS")

        print("\nRetrieved:")

        for hit in result["retrieved"]:
            marker = "*" if hit["chunk_id"] in result["relevant_chunk_ids"] else " "

            print(
                f"{marker} {hit['rank']}. "
                f"vector={hit['vector_score']:.3f} "
                f"rerank={hit['rerank_score']:.3f} "
                f"{hit['title']}\n"
                f"     {hit['chunk_id']}"
            )

    total = len(results)

    hit_at_1 = sum(
        result["hit_rank"] is not None and result["hit_rank"] <= 1
        for result in results
    )

    hit_at_3 = sum(
        result["hit_rank"] is not None and result["hit_rank"] <= 3
        for result in results
    )

    hit_at_5 = sum(
        result["hit_rank"] is not None and result["hit_rank"] <= 5
        for result in results
    )

    mrr = sum(
        1.0 / result["hit_rank"]
        if result["hit_rank"] is not None
        else 0.0
        for result in results
    ) / total

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"MRR:   {mrr:.3f}")

    print(f"Hit@1: {hit_at_1}/{total} = {hit_at_1 / total:.1%}")
    print(f"Hit@3: {hit_at_3}/{total} = {hit_at_3 / total:.1%}")
    print(f"Hit@5: {hit_at_5}/{total} = {hit_at_5 / total:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description = "Evaluate semantic retrieval quality"
    )

    parser.add_argument(
        "--corpus",
        default = "edrm",
        help = "Chroma collection to evaluate",
    )

    parser.add_argument(
        "-k",
        type = int,
        default = 5,
        help = "Number of retrieved chunks per query",
    )

    args = parser.parse_args()

    evaluate(
        corpus = args.corpus,
        k = args.k,
    )


if __name__ == "__main__":
    main()
