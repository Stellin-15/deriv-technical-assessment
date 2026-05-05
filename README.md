# Deriv Help Centre RAG Pipeline

A replayable retrieval-augmented generation system over Deriv help centre content. The pipeline ingests, chunks, embeds, retrieves, answers, verifies grounding, and suppresses low-confidence answers before returning them.

## Setup

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install anthropic python-dotenv requests beautifulsoup4 sentence-transformers numpy playwright

# Install Playwright browser
python -m playwright install chromium

# Create .env file
cp .env.example .env
# Add your Anthropic API key to .env
```

## Running

```powershell
# Run the full pipeline (scrape → chunk → embed → query → audit)
python pipeline.py

# Validate all artifacts
python validate.py

# Start the interactive multi-turn CLI
python cli.py
```

## Pipeline Stages

```
INIT
 → SOURCES_LOADED          reads sources.json and test_queries.json
 → CONTENT_SCRAPED         Playwright scrapes JS-rendered help pages
 → CORPUS_CHUNKED          splits text into 200-350 token chunks
 → EMBEDDINGS_CACHED       all-MiniLM-L6-v2 embeddings, hash-based cache
 → QUERY_RECEIVED          for each query in test_queries.json
 → CHUNKS_RETRIEVED        top-5 cosine similarity (pure numpy)
 → CONFIDENCE_CHECKED      deterministic 0.72 threshold — no LLM
 → ANSWER_GENERATED        Stage 1 LLM call (claude-sonnet-4-5)
 → GROUNDING_VERIFIED      Stage 2 LLM call — claim-level verification
 → ANSWER_REGENERATED_IF_NEEDED  targeted regeneration for ungrounded claims
 → QUALITY_SCORED          Stage 3 LLM call — completeness/specificity/tone
 → ANSWER_RETURNED_OR_FALLBACK
 → AUDIT_EXPORTED          full answer_audit.json
```

## Configuration

| File | Purpose |
|---|---|
| `sources.json` | URLs to scrape (replaceable by evaluator) |
| `test_queries.json` | Queries to run through pipeline (replaceable) |
| `.env` | `ANTHROPIC_API_KEY` |

## Generated Artifacts

All artifacts are regenerated on each `python pipeline.py` run:

| File | Stage | Description |
|---|---|---|
| `scraped_content.json` | s1 | Raw scraped text per URL |
| `corpus.json` | s2 | Chunked content with metadata |
| `vectors.json` | s3 | Cached embeddings (hash-based, incremental) |
| `corpus_version_report.json` | s3 | Chunk diff stats |
| `retrieval_logs.jsonl` | s4 | Per-query retrieval evidence |
| `generated_answers.json` | s6 | LLM-generated answers |
| `grounding_verification.json` | s7 | Claim-level grounding results |
| `answer_audit.json` | s8/s11 | Full audit trail per query |
| `answer_quality_scores.json` | s9 | Completeness/specificity/tone scores |
| `knowledge_gap_report.json` | s10 | Topic gap analysis |
| `llm_calls.jsonl` | all | Every LLM call logged |

## File Structure

```
├── pipeline.py              main entry point
├── cli.py                   multi-turn interactive CLI
├── validate.py              15-check validation script
├── sources.json             source URLs
├── test_queries.json        test queries
├── .env                     API keys (not committed)
└── stages/
    ├── s1_ingest.py         Playwright scraper
    ├── s2_chunk.py          200-350 token chunker
    ├── s3_embed.py          embedding + cache
    ├── s4_retrieve.py       cosine similarity retrieval
    ├── s5_confidence.py     deterministic confidence check
    ├── s6_answer.py         Stage 1 LLM — answer generation
    ├── s7_verify.py         Stage 2 LLM — grounding verification
    ├── s8_regenerate.py     targeted regeneration
    ├── s9_quality.py        Stage 3 LLM — quality scoring
    ├── s10_gap.py           gap detection
    └── s11_audit.py         audit export
```

## Key Design Decisions

- **Playwright** over requests — Deriv's help pages are JavaScript-rendered (React)
- **Confidence threshold is deterministic** — 0.72 similarity cutoff, no LLM involvement
- **Grounding verification is a separate LLM call** — not bundled with answer generation
- **Embeddings are cached by content hash** — never recomputed for unchanged chunks
- **Regeneration prompt differs from original** — explicitly lists ungrounded claims to avoid

## Validation

```powershell
python validate.py
```

Runs 15 checks and prints `PASS`/`FAIL` per check. Exit code 0 if all pass.
