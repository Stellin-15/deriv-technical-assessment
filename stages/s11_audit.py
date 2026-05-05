import json
import os

OUTPUT_FILE = "answer_audit.json"


def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return default


def _load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def export(query_results: list) -> list:
    """
    query_results: list of per-query result dicts accumulated by pipeline.py
    """
    generated = _load_json("generated_answers.json", [])
    gen_by_id = {r["query_id"]: r for r in generated}

    grounding = _load_json("grounding_verification.json", [])
    grounding_by_id = {}
    for r in grounding:
        qid = r["query_id"]
        grounding_by_id.setdefault(qid, []).append(r)

    regen_records = _load_json(OUTPUT_FILE, [])
    regen_by_id = {}
    for r in regen_records:
        if "regenerated_answer" in r:
            regen_by_id[r["query_id"]] = r

    quality = _load_json("answer_quality_scores.json", [])
    quality_by_id = {r["query_id"]: r for r in quality}

    retrieval_logs = _load_jsonl("retrieval_logs.jsonl")
    retrieval_by_id = {}
    for r in retrieval_logs:
        retrieval_by_id[r["query_id"]] = r.get("retrieved_chunks", [])

    audit = []
    for qr in query_results:
        qid = qr["query_id"]
        query_text = qr["query_text"]

        retrieved = retrieval_by_id.get(qid, [])
        highest_score = max((c["similarity_score"] for c in retrieved), default=0.0)
        fallback_triggered = qr.get("fallback_triggered", False)

        gen = gen_by_id.get(qid, {})
        generated_answer = gen.get("answer", "")

        regen = regen_by_id.get(qid)
        regeneration_triggered = regen is not None
        regeneration_details = regen if regen else {}

        final_response = qr.get("final_response", "")

        record = {
            "query_id": qid,
            "query_text": query_text,
            "conversation_context": qr.get("conversation_context", []),
            "retrieved_chunks": retrieved,
            "highest_similarity_score": highest_score,
            "fallback_triggered": fallback_triggered,
            "generated_answer": generated_answer,
            "grounding_verification": grounding_by_id.get(qid, []),
            "regeneration_triggered": regeneration_triggered,
            "regeneration_details": regeneration_details,
            "quality_scores": quality_by_id.get(qid, {}),
            "final_response": final_response,
        }
        audit.append(record)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print(f"AUDIT EXPORTED — {len(audit)} query records")
    return audit
