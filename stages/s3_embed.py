import json
import os
from datetime import datetime

from sentence_transformers import SentenceTransformer

CORPUS_FILE = "corpus.json"
VECTORS_FILE = "vectors.json"
REPORT_FILE = "corpus_version_report.json"
MODEL_NAME = "all-MiniLM-L6-v2"


def run(corpus_file: str = CORPUS_FILE) -> list:
    with open(corpus_file, encoding="utf-8") as f:
        corpus = json.load(f)
    chunks = corpus["chunks"]

    # Load existing cache keyed by content_hash.
    cache: dict = {}
    if os.path.exists(VECTORS_FILE):
        with open(VECTORS_FILE, encoding="utf-8") as f:
            existing = json.load(f)
        for rec in existing:
            cache[rec["content_hash"]] = rec

    existing_hashes = set(cache.keys())
    current_hashes = {c["content_hash"] for c in chunks}

    chunks_unchanged = 0
    chunks_updated = 0
    chunks_added = 0
    chunks_removed = len(existing_hashes - current_hashes)

    to_embed = []
    for chunk in chunks:
        h = chunk["content_hash"]
        if h in cache:
            chunks_unchanged += 1
        else:
            # Determine added vs updated (chunk_id already existed with different hash).
            if any(
                rec.get("chunk_id") == chunk["chunk_id"]
                for rec in cache.values()
            ):
                chunks_updated += 1
            else:
                chunks_added += 1
            to_embed.append(chunk)

    if to_embed:
        print(f"  Loading model {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME)
        texts = [c["text"] for c in to_embed]
        print(f"  Embedding {len(texts)} new/changed chunks...")
        embeddings = model.encode(texts, show_progress_bar=True)
        for chunk, emb in zip(to_embed, embeddings):
            cache[chunk["content_hash"]] = {
                "chunk_id": chunk["chunk_id"],
                "content_hash": chunk["content_hash"],
                "embedding": emb.tolist(),
                "source_url": chunk["source_url"],
                "text": chunk["text"],
                "section_title": chunk["section_title"],
            }

    # Build final vector list preserving only current chunks.
    vectors = [cache[c["content_hash"]] for c in chunks if c["content_hash"] in cache]

    with open(VECTORS_FILE, "w", encoding="utf-8") as f:
        json.dump(vectors, f, ensure_ascii=False)

    report = {
        "computed_at": datetime.utcnow().isoformat() + "Z",
        "chunks_unchanged": chunks_unchanged,
        "chunks_updated": chunks_updated,
        "chunks_added": chunks_added,
        "chunks_removed": chunks_removed,
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    n_embedded = len(to_embed)
    n_cached = chunks_unchanged
    print(f"STAGE: EMBEDDINGS_CACHED — {n_embedded} embedded, {n_cached} cached")
    return vectors


if __name__ == "__main__":
    run()
