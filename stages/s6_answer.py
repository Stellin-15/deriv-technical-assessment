import hashlib
import json
import os
from datetime import datetime

import anthropic
from dotenv import load_dotenv

load_dotenv()

OUTPUT_FILE = "generated_answers.json"
LLM_LOG = "llm_calls.jsonl"
MODEL = "claude-sonnet-4-5"


def _build_prompt(query: str, chunks: list) -> tuple[str, str]:
    system = (
        "You are a Deriv Help Centre assistant. Answer the user's question using ONLY "
        "the provided context chunks. Do not add any information not present in the chunks. "
        "If the context does not contain the answer, say explicitly: "
        "'I could not find this information in the Help Centre content.' "
        "Always cite the chunk IDs you used in your answer using format [chunk_id]."
    )

    chunk_text = ""
    for c in chunks:
        chunk_text += (
            f"CHUNK ID: {c['chunk_id']}\n"
            f"SOURCE: {c['source_url']}\n\n"
            f"{c['text']}\n\n---\n\n"
        )

    user = (
        f"Query: {query}\n\n"
        f"Context chunks:\n{chunk_text}"
        f"Answer the query based only on the above context. Cite chunk IDs."
    )

    return system, user


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


def generate(query: str, query_id: str, chunks: list) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    system, user = _build_prompt(query, chunks)

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    answer = response.content[0].text

    # Persist answer.
    record = {
        "query_id": query_id,
        "query_text": query,
        "answer": answer,
        "chunks_used": [c["chunk_id"] for c in chunks],
    }
    _append_json(OUTPUT_FILE, record)

    _log_llm_call(
        "answer_generation", query_id, system + user,
        ["corpus.json", "vectors.json"], OUTPUT_FILE,
    )

    print(f"ANSWER GENERATED for {query_id}")
    return answer


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
