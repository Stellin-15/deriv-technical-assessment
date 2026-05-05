CONFIDENCE_THRESHOLD = 0.72


def check(retrieved_chunks: list) -> dict:
    """Deterministic confidence check — no LLM involved."""
    if not retrieved_chunks:
        result = {
            "fallback": True,
            "reason": "No content was retrieved from the indexed Help Centre.",
            "best_source_url": None,
            "similarity_score": 0.0,
            "suggested_action": "Please visit the Deriv Help Centre directly at https://deriv.com/help-centre/",
        }
        print(f"CONFIDENCE CHECK: score=0.0000, fallback=true")
        return result

    highest_score = max(c["similarity_score"] for c in retrieved_chunks)
    best_chunk = max(retrieved_chunks, key=lambda c: c["similarity_score"])

    if highest_score < CONFIDENCE_THRESHOLD:
        result = {
            "fallback": True,
            "reason": "No sufficiently relevant content found in the indexed Help Centre.",
            "best_source_url": best_chunk["source_url"],
            "similarity_score": highest_score,
            "suggested_action": "Please visit the Deriv Help Centre directly at https://deriv.com/help-centre/",
        }
        print(f"CONFIDENCE CHECK: score={highest_score:.4f}, fallback=true")
    else:
        result = {
            "fallback": False,
            "confidence_score": highest_score,
        }
        print(f"CONFIDENCE CHECK: score={highest_score:.4f}, fallback=false")

    return result
