import hashlib
import json
import os
import re
from datetime import datetime

import anthropic
from dotenv import load_dotenv

load_dotenv()

OUTPUT_FILE = "knowledge_gap_report.json"
LLM_LOG = "llm_calls.jsonl"
MODEL = "claude-sonnet-4-5"
GAP_THRESHOLD = 0.80


def _log_llm_call(stage: str, query_id: str, prompt: str,
                  input_artifacts: list, output_artifact: str):
    record = {
        "stage": stage,
        "query_id": query_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "provider": "anthropic",
        "model": MODEL,
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
        "input_artifacts": input_artifacts,
        "output_artifact": output_artifact,
    }
    with open(LLM_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def detect(query_results: list) -> list:
    """
    query_results: list of {query_id, query_text, highest_score, fallback_triggered}
    """
    weak = [
        r for r in query_results
        if r.get("highest_score", 0.0) < GAP_THRESHOLD or r.get("fallback_triggered", False)
    ]

    if not weak:
        print("GAP DETECTION COMPLETE — 0 gaps identified")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return []

    lines = "\n".join(
        f"- [{r['query_id']}] \"{r['query_text']}\" (score={r['highest_score']:.4f})"
        for r in weak
    )

    system = "You are a knowledge gap analyst for a help centre RAG system."
    user = (
        "These queries returned weak retrieval results from the Deriv Help Centre:\n"
        f"{lines}\n\n"
        "Group them into topic categories and suggest what content is missing.\n"
        "Respond with JSON array:\n"
        "[\n"
        "  {\n"
        '    "topic": "string",\n'
        '    "query_ids": ["Q1"],\n'
        '    "evidence": "string",\n'
        '    "recommended_content_improvement": "string"\n'
        "  }\n"
        "]"
    )

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    text = response.content[0].text
    text = re.sub(r"```(?:json)?\n?", "", text).strip().rstrip("`").strip()

    try:
        gaps = json.loads(text)
    except json.JSONDecodeError:
        gaps = [{
            "topic": "Parse error",
            "query_ids": [r["query_id"] for r in weak],
            "evidence": "LLM returned non-JSON",
            "recommended_content_improvement": "Manual review required",
        }]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(gaps, f, indent=2, ensure_ascii=False)

    _log_llm_call(
        "gap_detection", None, system + user,
        ["retrieval_logs.jsonl"], OUTPUT_FILE,
    )

    print(f"GAP DETECTION COMPLETE — {len(gaps)} gaps identified")
    return gaps
