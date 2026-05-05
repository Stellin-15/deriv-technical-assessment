"""
Validation script — checks all required artifacts exist and are correctly structured.
Run: python validate.py
Exit code 0 if all checks pass, 1 if any fail.
"""
import json
import os
import sys


PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    msg = f"[{status}] {name}"
    if detail and not condition:
        msg += f" — {detail}"
    print(msg)
    results.append(condition)


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None


def load_jsonl(path: str):
    if not os.path.exists(path):
        return None
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                return None
    return records


def main():
    print("=" * 60)
    print("  Deriv RAG Pipeline — Validation")
    print("=" * 60)

    # 1. sources.json exists and valid JSON
    sources = load_json("sources.json")
    check("1. sources.json exists and valid JSON",
          sources is not None and "sources" in sources)

    # 2. test_queries.json exists and valid JSON
    queries_raw = load_json("test_queries.json")
    check("2. test_queries.json exists and valid JSON",
          isinstance(queries_raw, list) and len(queries_raw) > 0)
    query_ids = [q["id"] for q in (queries_raw or []) if "id" in q]

    # 3. corpus.json exists and has chunks
    corpus = load_json("corpus.json")
    chunks = corpus.get("chunks", []) if isinstance(corpus, dict) else []
    check("3. corpus.json exists and has chunks",
          corpus is not None and len(chunks) > 0,
          f"found {len(chunks)} chunks")

    # 4. vectors.json exists and has embeddings
    vectors = load_json("vectors.json")
    check("4. vectors.json exists and has embeddings",
          isinstance(vectors, list) and len(vectors) > 0,
          f"found {len(vectors) if vectors else 0} vectors")

    # 5. All chunk_ids in corpus have embedding in vectors.json
    if chunks and vectors:
        chunk_ids = {c["chunk_id"] for c in chunks}
        vector_chunk_ids = {v["chunk_id"] for v in vectors}
        missing = chunk_ids - vector_chunk_ids
        check("5. All chunk_ids in corpus have embedding in vectors.json",
              len(missing) == 0,
              f"{len(missing)} missing: {list(missing)[:3]}")
    else:
        check("5. All chunk_ids in corpus have embedding in vectors.json",
              False, "corpus or vectors empty")

    # 6. generated_answers.json exists
    answers = load_json("generated_answers.json")
    check("6. generated_answers.json exists",
          isinstance(answers, list),
          "file missing or invalid")

    # 7. grounding_verification.json exists
    grounding = load_json("grounding_verification.json")
    check("7. grounding_verification.json exists",
          isinstance(grounding, list),
          "file missing or invalid")

    # 8. answer_audit.json exists
    audit = load_json("answer_audit.json")
    check("8. answer_audit.json exists",
          isinstance(audit, list) or isinstance(audit, dict),
          "file missing or invalid")

    # 9. llm_calls.jsonl exists
    llm_calls = load_jsonl("llm_calls.jsonl")
    check("9. llm_calls.jsonl exists",
          isinstance(llm_calls, list),
          "file missing or invalid")

    # 10. llm_calls.jsonl has answer_generation record per non-fallback query
    if llm_calls and answers:
        answered_ids = {a["query_id"] for a in answers}
        gen_ids = {r["query_id"] for r in llm_calls if r.get("stage") == "answer_generation"}
        missing_gen = answered_ids - gen_ids
        check("10. llm_calls.jsonl has answer_generation record per answered query",
              len(missing_gen) == 0,
              f"missing for: {missing_gen}")
    else:
        check("10. llm_calls.jsonl has answer_generation record per answered query",
              True, "no answers generated (all fallback) or no LLM calls")

    # 11. llm_calls.jsonl has grounding_verification record per answered query
    if llm_calls and answers:
        verify_ids = {r["query_id"] for r in llm_calls
                      if r.get("stage") == "grounding_verification"}
        missing_verify = answered_ids - verify_ids
        check("11. llm_calls.jsonl has grounding_verification record per answered query",
              len(missing_verify) == 0,
              f"missing for: {missing_verify}")
    else:
        check("11. llm_calls.jsonl has grounding_verification record per answered query",
              True, "no answers generated (all fallback)")

    # 12. All fallback decisions reference a similarity score
    if audit and isinstance(audit, list):
        fallback_records = [r for r in audit if r.get("fallback_triggered")]
        bad_fallbacks = [r for r in fallback_records
                         if "highest_similarity_score" not in r]
        check("12. All fallback decisions reference a similarity score",
              len(bad_fallbacks) == 0,
              f"{len(bad_fallbacks)} fallback records missing similarity score")
    else:
        check("12. All fallback decisions reference a similarity score",
              True, "no audit records")

    # 13. retrieval_logs.jsonl exists
    retrieval_logs = load_jsonl("retrieval_logs.jsonl")
    check("13. retrieval_logs.jsonl exists",
          isinstance(retrieval_logs, list),
          "file missing or invalid")

    # 14. answer_quality_scores.json exists
    quality = load_json("answer_quality_scores.json")
    check("14. answer_quality_scores.json exists",
          isinstance(quality, list),
          "file missing or invalid")

    # 15. knowledge_gap_report.json exists
    gaps = load_json("knowledge_gap_report.json")
    check("15. knowledge_gap_report.json exists",
          isinstance(gaps, list),
          "file missing or invalid")

    # Summary
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print()
    print("=" * 60)
    print(f"  {n_pass} checks passed, {n_fail} failed")
    print("=" * 60)

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
