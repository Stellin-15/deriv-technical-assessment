import hashlib
import json
import re
from datetime import datetime
from urllib.parse import urlparse

INPUT_FILE = "scraped_content.json"
OUTPUT_FILE = "corpus.json"

CHUNK_TARGET = 250
CHUNK_OVERLAP = 50
CHUNK_MIN = 200
CHUNK_MAX = 350


def url_to_slug(url: str) -> str:
    parsed = urlparse(url)
    slug = parsed.netloc + parsed.path
    slug = re.sub(r"[^a-zA-Z0-9]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:80]


def extract_section_title(words_before: list, page_title: str) -> str:
    text_before = " ".join(words_before[-100:]) if words_before else ""
    lines = [l.strip() for l in text_before.split("\n") if l.strip()]
    for line in reversed(lines):
        word_count = len(line.split())
        if 1 <= word_count <= 10 and not line.endswith((".", "?", "!")):
            return line
    return page_title


def chunk_text(text: str, slug: str, source_url: str) -> list:
    words = text.split()
    chunks = []
    start = 0
    index = 0

    while start < len(words):
        end = min(start + CHUNK_TARGET, len(words))
        chunk_words = words[start:end]

        if len(chunk_words) < CHUNK_MIN and chunks:
            last = chunks[-1]
            last["text"] = last["text"] + " " + " ".join(chunk_words)
            last["token_count"] = len(last["text"].split())
            last["content_hash"] = hashlib.sha256(last["text"].encode()).hexdigest()
            break

        chunk_text_str = " ".join(chunk_words)
        content_hash = hashlib.sha256(chunk_text_str.encode()).hexdigest()
        section_title = extract_section_title(words[:start], slug.replace("_", " "))

        chunks.append({
            "chunk_id": f"{slug}_chunk_{index}",
            "source_url": source_url,
            "section_title": section_title,
            "chunk_index": index,
            "token_count": len(chunk_words),
            "content_hash": content_hash,
            "text": chunk_text_str,
        })

        index += 1
        start = end - CHUNK_OVERLAP

    return chunks


def run(input_file: str = INPUT_FILE) -> list:
    with open(input_file, encoding="utf-8") as f:
        pages = json.load(f)

    all_chunks = []
    for page in pages:
        if not page.get("success") or not page.get("raw_text"):
            continue
        slug = url_to_slug(page["url"])
        chunks = chunk_text(page["raw_text"], slug, page["url"])
        all_chunks.extend(chunks)

    corpus = {
        "corpus_version": datetime.utcnow().isoformat() + "Z",
        "chunks": all_chunks,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    print(f"STAGE: CORPUS_CHUNKED — {len(all_chunks)} chunks created")
    return all_chunks


if __name__ == "__main__":
    run()
