"""Embed chunks.jsonl into a local Chroma index and run queries.

Build:
    C:\\Users\\gokif\\anaconda3\\python.exe src\\index.py

Query:
    C:\\Users\\gokif\\anaconda3\\python.exe src\\index.py query "information governance"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "index"
COLLECTION = "edrm"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}\nrun ingest.py first")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no chunks in {path}")
    return rows


def flatten_metadata(chunk: dict) -> dict:
    extra = chunk.get("metadata") or {}
    meta = {
        "doc_id": str(chunk.get("doc_id") or ""),
        "source_path": str(chunk.get("source_path") or ""),
        "file_type": str(chunk.get("file_type") or ""),
        "title": str(chunk.get("title") or ""),
        "chunk_index": int(chunk.get("chunk_index") or 0),
        "from": str(extra.get("from") or ""),
        "to": str(extra.get("to") or ""),
        "date": str(extra.get("date") or ""),
        "subject": str(extra.get("subject") or ""),
        "attachments": json.dumps(extra.get("attachments") or [], ensure_ascii=False),
    }
    return meta


def collection(reset: bool = False):
    import chromadb

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def build() -> None:
    chunks = load_chunks(CHUNKS_PATH)
    model = embedder()
    coll = collection(reset=True)

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [flatten_metadata(c) for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    coll.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=vectors.tolist(),
    )
    print(f"indexed {len(chunks)} chunks -> {INDEX_DIR}")
    print(f"collection: {COLLECTION}")
    print(f"model: {MODEL_NAME}")


def query(text: str, k: int = 5) -> None:
    model = embedder()
    coll = collection(reset=False)
    if coll.count() == 0:
        raise SystemExit("empty index; run: python src/index.py")

    vector = model.encode([text], normalize_embeddings=True)[0].tolist()
    result = coll.query(
        query_embeddings=[vector],
        n_results=min(k, coll.count()),
        include=["documents", "metadatas", "distances"],
    )
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    print(f"query: {text}")
    print(f"hits: {len(docs)}")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        score = 1.0 - float(dist)
        print(f"\n[{i}] score={score:.3f}  {meta.get('file_type')}  {meta.get('title')}")
        print(f"    {meta.get('source_path')}")
        if meta.get("from"):
            print(f"    from: {meta.get('from')}")
        preview = (doc or "").replace("\n", " ")
        print(f"    {preview[:400]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query the EDRM Chroma index")
    parser.add_argument("command", nargs="?", default="build", choices=["build", "query"])
    parser.add_argument("text", nargs="?", default="information governance digital debris")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    if args.command == "query":
        query(args.text, k=args.k)
    else:
        build()


if __name__ == "__main__":
    main()
