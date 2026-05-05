import hashlib
import json
import os
import re
from datetime import datetime

import anthropic
from dotenv import load_dotenv

load_dotenv()

OUTPUT_FILE = "answer_quality_scores.json"
LLM_LOG = "llm_calls.jsonl"
MODEL = "claude-sonnet-4-5"


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


def _append_json(path: str, record: dict):
    existing = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    existing.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def score(answer: str, query: str, query_id: str) -> dict:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    system = (
        "Score the following answer on three dimensions. Respond with JSON only."
    )
    user = (
        f"Answer: {answer}\n"
        f"Query: {query}\n\n"
        "Score on:\n"
        "- completeness: does it fully address the question? 0-10\n"
        "- specificity: does it give specific details not vague statements? 0-10\n"
        "- tone: is it helpful, clear, and appropriate? 0-10\n\n"
        "Respond with:\n"
        '{"completeness": 8, "specificity": 7, "tone": 9, "flags": [], "overall": 8}'
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    text = response.content[0].text
    text = re.sub(r"```(?:json)?\n?", "", text).strip().rstrip("`").strip()

    try:
        scores = json.loads(text)
    except json.JSONDecodeError:
        scores = {"completeness": 0, "specificity": 0, "tone": 0, "flags": ["parse_error"], "overall": 0}

    # Flag dimensions below 6.
    flags = scores.get("flags", [])
    for dim in ["completeness", "specificity", "tone"]:
        if scores.get(dim, 10) < 6:
            flag = f"{dim}_below_threshold"
            if flag not in flags:
                flags.append(flag)
    scores["flags"] = flags
    scores["query_id"] = query_id

    _append_json(OUTPUT_FILE, scores)

    _log_llm_call(
        "quality_scoring", query_id, system + user,
        ["generated_answers.json", "grounding_verification.json"], OUTPUT_FILE,
    )

    c = scores.get("completeness", 0)
    s = scores.get("specificity", 0)
    t = scores.get("tone", 0)
    print(f"QUALITY SCORED — completeness={c}, specificity={s}, tone={t}")
    return scores
