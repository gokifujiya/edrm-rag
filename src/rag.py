from pathlib import Path
import re

import chromadb
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index"

COLLECTION = "edrm"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

DENSE_K = 10
BM25_K = 10
RRF_K = 60
FINAL_RESULTS = 5

OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "llama3.2"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def reciprocal_rank_fusion(
    dense_ids: list[str],
    bm25_ids: list[str],
    rrf_k: int = RRF_K,
) -> list[str]:
    scores = {}

    for rank, chunk_id in enumerate(dense_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (
            rrf_k + rank
        )

    for rank, chunk_id in enumerate(bm25_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (
            rrf_k + rank
        )

    return sorted(
        scores,
        key=scores.get,
        reverse=True,
    )


def retrieve(question: str, n_results: int = 10):
    """Retrieve EDRM chunks using dense search + BM25 + RRF."""

    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    collection = client.get_collection(COLLECTION)

    # ---------------------------------------------------------
    # Dense retrieval
    # ---------------------------------------------------------

    query_embedding = model.encode(
        [question],
        normalize_embeddings = True,
    )[0].tolist()

    dense_result = collection.query(
        query_embeddings = [query_embedding],
        n_results = min(DENSE_K, collection.count()),
        include = ["documents", "metadatas", "distances"],
    )

    dense_ids = dense_result["ids"][0]

    dense_scores = {
        chunk_id: 1.0 - float(distance)
        for chunk_id, distance in zip(
            dense_ids,
            dense_result["distances"][0],
        )
    }

    # ---------------------------------------------------------
    # Load corpus for BM25
    # ---------------------------------------------------------

    corpus_data = collection.get(
        include=["documents", "metadatas"],
    )

    corpus_ids = corpus_data["ids"]
    corpus_documents = corpus_data["documents"]
    corpus_metadatas = corpus_data["metadatas"]

    documents_by_id = {
        chunk_id: document
        for chunk_id, document in zip(
            corpus_ids,
            corpus_documents,
        )
    }

    metadata_by_id = {
        chunk_id: metadata
        for chunk_id, metadata in zip(
            corpus_ids,
            corpus_metadatas,
        )
    }

    # ---------------------------------------------------------
    # BM25 retrieval
    # ---------------------------------------------------------

    tokenized_corpus = [
        tokenize(document)
        for document in corpus_documents
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    bm25_scores = bm25.get_scores(
        tokenize(question)
    )

    bm25_ranked_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:BM25_K]

    bm25_ids = [
        corpus_ids[i]
        for i in bm25_ranked_indices
    ]

    bm25_score_by_id = {
        corpus_ids[i]: float(bm25_scores[i])
        for i in bm25_ranked_indices
    }

    # ---------------------------------------------------------
    # Reciprocal Rank Fusion
    # ---------------------------------------------------------

    fused_ids = reciprocal_rank_fusion(
        dense_ids,
        bm25_ids,
    )

    hits = []

    for chunk_id in fused_ids:
        hits.append(
            {
                "chunk_id": chunk_id,
                "text": documents_by_id[chunk_id],
                "metadata": metadata_by_id[chunk_id],
                "score": dense_scores.get(chunk_id),
                "bm25_score": bm25_score_by_id.get(chunk_id),
            }
        )

    return hits[:n_results]


def rerank(question: str, hits: list[dict], n_results: int = 5):
    """Rerank retrieved chunks using a cross-encoder."""

    model = CrossEncoder(RERANK_MODEL)

    pairs = [
        [question, hit["text"]]
        for hit in hits
    ]

    scores = model.predict(pairs)

    for hit, score in zip(hits, scores):
        hit["retrieval_score"] = hit["score"]
        hit["rerank_score"] = float(score)

    hits.sort(
        key = lambda hit: hit["rerank_score"],
        reverse = True,
    )

    return hits[:n_results]


def build_prompt(question: str, hits: list[dict]) -> str:
    """Create a grounded prompt from retrieved EDRM chunks."""

    sources = []

    for number, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]
        source_path = metadata.get("source_path", "Unknown source")

        sources.append(
            f"""SOURCE {number}
File: {source_path}
Content:
{hit["text"]}"""
        )

    context = "\n\n".join(sources)

    return f"""
Answer the question using only the EDRM sources below.

If the sources do not contain enough information to answer the
question, say that the available EDRM sources do not provide enough
information. Do not fill missing information from your own knowledge.

Cite supporting sources in the answer as [Source 1], [Source 2], etc.

EDRM SOURCES:

{context}

QUESTION:
{question}
"""


def ask_ollama(prompt: str) -> str:
    """Send the grounded prompt to the local Ollama model."""

    response = requests.post(
        OLLAMA_URL,
        json = {
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
        },
        timeout = 300,
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


def main():
    question = input("Question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    print("\nSearching EDRM documents...")

    candidates = retrieve(
        question,
        n_results = DENSE_K + BM25_K,
    )

    print("Reranking retrieved chunks...")

    hits = rerank(
        question,
        candidates,
        n_results = FINAL_RESULTS,
    )

    prompt = build_prompt(question, hits)

    print("Generating answer with Ollama...\n")
    answer = ask_ollama(prompt)

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for number, hit in enumerate(hits, start = 1):
        metadata = hit["metadata"]

        print(
            f"[Source {number}] "
            f"vector={hit['retrieval_score']:.3f}  "
            f"rerank={hit['rerank_score']:.3f}\n"
            f"  chunk={hit['chunk_id']}\n"
            f"  file={metadata.get('source_path', 'Unknown source')}"
        )


if __name__ == "__main__":
    main()
