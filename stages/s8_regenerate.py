import hashlib
import json
import os
from datetime import datetime

import anthropic
from dotenv import load_dotenv

from stages.s7_verify import verify

load_dotenv()

AUDIT_FILE = "answer_audit.json"
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


def _build_regen_prompt(query: str, chunks: list, ungrounded_claims: list) -> tuple[str, str]:
    system = (
        "You are a Deriv Help Centre assistant. Answer the user's question using ONLY "
        "the provided context chunks. Do not add any information not present in the chunks. "
        "If the context does not contain the answer, say explicitly: "
        "'I could not find this information in the Help Centre content.' "
        "Always cite the chunk IDs you used in your answer using format [chunk_id]."
    )

    claims_list = "\n".join(f"- {c}" for c in ungrounded_claims)
    chunk_text = ""
    for c in chunks:
        chunk_text += (
            f"CHUNK ID: {c['chunk_id']}\n"
            f"SOURCE: {c['source_url']}\n\n"
            f"{c['text']}\n\n---\n\n"
        )

    user = (
        f"The previous answer contained ungrounded claims. "
        f"Do NOT include the following claims as they are not supported by the source material:\n"
        f"{claims_list}\n\n"
        f"Query: {query}\n\n"
        f"Context chunks:\n{chunk_text}"
        f"Answer the query based only on the above context. "
        f"Cite chunk IDs. Strictly avoid any claims not present in the chunks."
    )
    return system, user


def regenerate(query: str, query_id: str, original_answer: str,
               chunks: list, claims: list) -> tuple[str, list]:
    """
    Returns (final_answer, second_verification_claims).
    If no ungrounded claims, returns original answer unchanged.
    """
    ungrounded = [c["claim"] for c in claims if not c.get("grounded", True)]

    if not ungrounded:
        return original_answer, claims

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    system, user = _build_regen_prompt(query, chunks, ungrounded)
    prompt_str = system + user

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    regenerated_answer = response.content[0].text

    _log_llm_call(
        "regeneration", query_id, prompt_str,
        ["corpus.json", "vectors.json", "grounding_verification.json"], AUDIT_FILE,
    )

    # Second grounding verification on the regenerated answer.
    second_claims = verify(regenerated_answer, query_id, chunks,
                           stage="second_verification")

    audit_record = {
        "query_id": query_id,
        "original_answer": original_answer,
        "ungrounded_claims": ungrounded,
        "regeneration_prompt_hash": hashlib.sha256(prompt_str.encode()).hexdigest(),
        "regenerated_answer": regenerated_answer,
        "second_verification": second_claims,
    }
    _append_json(AUDIT_FILE, audit_record)

    n_removed = len(ungrounded)
    print(f"REGENERATION COMPLETE — {n_removed} ungrounded claims removed")
    return regenerated_answer, second_claims
