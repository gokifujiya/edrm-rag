from pathlib import Path

import chromadb
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder


ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index"

COLLECTION = "edrm"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

RETRIEVAL_CANDIDATES = 10
FINAL_RESULTS = 5

OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "llama3.2"


def retrieve(question: str, n_results: int = 5):
    """Find the most relevant EDRM chunks."""

    model = SentenceTransformer(EMBEDDING_MODEL)

    query_embedding = model.encode(
        [question],
        normalize_embeddings = True,
    )[0].tolist()

    client = chromadb.PersistentClient(path = str(INDEX_DIR))
    collection = client.get_collection(COLLECTION)

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results = n_results,
        include = ["documents", "metadatas", "distances"],
    )

    hits = []

    for chunk_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "chunk_id": chunk_id,
                "text": document,
                "metadata": metadata,
                "score": 1 - distance,
            }
        )

    return hits


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
        n_results = RETRIEVAL_CANDIDATES,
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
