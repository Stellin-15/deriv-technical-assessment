import json
import os
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer

VECTORS_FILE = "vectors.json"
RETRIEVAL_LOG = "retrieval_logs.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_vectors = None
_matrix = None


def _load(vectors_file: str = VECTORS_FILE):
    global _model, _vectors, _matrix
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    if _vectors is None:
        with open(vectors_file, encoding="utf-8") as f:
            _vectors = json.load(f)
        if _vectors:
            _matrix = np.array([v["embedding"] for v in _vectors], dtype=np.float32)
        else:
            _matrix = np.empty((0,), dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def retrieve(query: str, query_id: str, top_k: int = 5,
             vectors_file: str = VECTORS_FILE) -> list:
    _load(vectors_file)

    query_emb = _model.encode([query])[0].astype(np.float32)

    if _matrix is None or _matrix.shape[0] == 0:
        print("Retrieved 0 chunks, top score: 0.0000")
        return []

    # Compute cosine similarity for all chunks.
    norms = np.linalg.norm(_matrix, axis=1) * np.linalg.norm(query_emb) + 1e-10
    scores = _matrix.dot(query_emb) / norms

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        vec = _vectors[idx]
        results.append({
            "chunk_id": vec["chunk_id"],
            "source_url": vec["source_url"],
            "similarity_score": float(scores[idx]),
            "text": vec["text"],
            "section_title": vec.get("section_title", ""),
        })

    top_score = results[0]["similarity_score"] if results else 0.0
    print(f"Retrieved {len(results)} chunks, top score: {top_score:.4f}")

    # Log retrieval.
    log_record = {
        "query_id": query_id,
        "query_text": query,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "retrieved_chunks": results,
    }
    with open(RETRIEVAL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_record) + "\n")

    return results


def reset_cache():
    """Call between pipeline runs to force fresh load."""
    global _model, _vectors, _matrix
    _vectors = None
    _matrix = None


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "How do I reset 2FA?"
    results = retrieve(q, "TEST")
    for r in results:
        print(f"  [{r['similarity_score']:.4f}] {r['chunk_id']}")
