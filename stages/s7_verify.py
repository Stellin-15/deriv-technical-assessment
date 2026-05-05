import hashlib
import json
import os
import re
from datetime import datetime

import anthropic
from dotenv import load_dotenv

load_dotenv()

OUTPUT_FILE = "grounding_verification.json"
LLM_LOG = "llm_calls.jsonl"
MODEL = "claude-sonnet-4-5"


def _build_prompt(answer: str, chunks: list) -> tuple[str, str]:
    system = (
        "You are a grounding verifier. Your job is to check every factual claim in the answer "
        "against the provided source chunks. For each claim, determine if it is directly "
        "supported by a specific chunk."
    )

    chunk_text = ""
    for c in chunks:
        chunk_text += f"CHUNK ID: {c['chunk_id']}\n{c['text']}\n\n---\n\n"

    user = (
        f"Answer to verify:\n{answer}\n\n"
        f"Source chunks used:\n{chunk_text}"
        "For every factual claim in the answer, output a JSON array:\n"
        "[\n"
        "  {\n"
        '    "claim": "exact claim text",\n'
        '    "grounded": true,\n'
        '    "supporting_chunk_ids": ["chunk_id"],\n'
        '    "explanation": "found in chunk X which states..."\n'
        "  }\n"
        "]\n"
        "Mark grounded false if claim cannot be traced to any chunk.\n"
        "Respond with valid JSON array only."
    )
    return system, user


def _parse_json_response(text: str) -> list:
    text = re.sub(r"```(?:json)?\n?", "", text).strip().rstrip("`").strip()
    return json.loads(text)


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


def verify(answer: str, query_id: str, chunks: list,
           stage: str = "grounding_verification") -> list:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    system, user = _build_prompt(answer, chunks)

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    try:
        claims = _parse_json_response(response.content[0].text)
    except (json.JSONDecodeError, IndexError):
        claims = [{
            "claim": "Unable to parse verification response",
            "grounded": False,
            "supporting_chunk_ids": [],
            "explanation": "Verifier returned non-JSON response",
        }]

    record = {"query_id": query_id, "claims": claims}
    if stage == "grounding_verification":
        _append_json(OUTPUT_FILE, record)

    _log_llm_call(
        stage, query_id, system + user,
        [OUTPUT_FILE], OUTPUT_FILE,
    )

    n_ungrounded = sum(1 for c in claims if not c.get("grounded", True))
    print(f"GROUNDING VERIFIED — {len(claims)} claims, {n_ungrounded} ungrounded")
    return claims
