"""
Main pipeline entry point.
Run: python pipeline.py
"""
import json
import sys
import traceback

from stages import s1_ingest, s2_chunk, s3_embed, s4_retrieve, s5_confidence
from stages import s6_answer, s7_verify, s8_regenerate, s9_quality, s10_gap, s11_audit


def state(name: str):
    print(f"\n{'='*60}")
    print(f"  STATE: {name}")
    print(f"{'='*60}")


def run():
    try:
        state("INIT")

        # ── Stage 1: Load sources ──────────────────────────────────
        state("SOURCES_LOADED")
        with open("sources.json") as f:
            sources_data = json.load(f)
        with open("test_queries.json") as f:
            queries = json.load(f)
        print(f"Loaded {len(sources_data['sources'])} sources, {len(queries)} queries")

        # ── Stage 2: Scrape ────────────────────────────────────────
        state("CONTENT_SCRAPED")
        s1_ingest.run("sources.json")

        # ── Stage 3: Chunk ─────────────────────────────────────────
        state("CORPUS_CHUNKED")
        chunks = s2_chunk.run("scraped_content.json")

        # ── Stage 4: Embed ─────────────────────────────────────────
        state("EMBEDDINGS_CACHED")
        if not chunks:
            print("[WARN] No chunks to embed — corpus is empty. "
                  "Scraping likely returned no content.")
        s3_embed.run("corpus.json")

        # Clear retrieval cache so we load fresh vectors.
        s4_retrieve.reset_cache()

        # ── Per-query stages ───────────────────────────────────────
        query_results = []
        gap_inputs = []

        for q in queries:
            query_id = q["id"]
            query_text = q["query"]

            state(f"QUERY_RECEIVED — {query_id}")
            print(f"Query: {query_text}")

            # Retrieve.
            state("CHUNKS_RETRIEVED")
            retrieved = s4_retrieve.retrieve(query_text, query_id)

            highest_score = (
                max(c["similarity_score"] for c in retrieved)
                if retrieved else 0.0
            )

            # Confidence check.
            state("CONFIDENCE_CHECKED")
            confidence = s5_confidence.check(retrieved)

            query_result = {
                "query_id": query_id,
                "query_text": query_text,
                "conversation_context": [],
                "fallback_triggered": confidence["fallback"],
                "final_response": "",
            }

            if confidence["fallback"]:
                state("ANSWER_RETURNED_OR_FALLBACK")
                fallback_msg = (
                    f"I was unable to find a confident answer in the Deriv Help Centre "
                    f"for your query.\n\n"
                    f"Reason: {confidence['reason']}\n"
                    f"Best matching source: {confidence.get('best_source_url', 'N/A')}\n"
                    f"Similarity score: {confidence.get('similarity_score', 0.0):.4f}\n\n"
                    f"{confidence.get('suggested_action', '')}"
                )
                query_result["final_response"] = fallback_msg
                print(f"FALLBACK triggered for {query_id}")
            else:
                # Answer generation.
                state("ANSWER_GENERATED")
                answer = s6_answer.generate(query_text, query_id, retrieved)

                # Grounding verification.
                state("GROUNDING_VERIFIED")
                claims = s7_verify.verify(answer, query_id, retrieved)

                # Regeneration if needed.
                state("ANSWER_REGENERATED_IF_NEEDED")
                final_answer, final_claims = s8_regenerate.regenerate(
                    query_text, query_id, answer, retrieved, claims
                )

                # Quality scoring.
                state("QUALITY_SCORED")
                s9_quality.score(final_answer, query_text, query_id)

                state("ANSWER_RETURNED_OR_FALLBACK")
                query_result["final_response"] = final_answer
                print(f"\nFINAL ANSWER for {query_id}:\n{final_answer[:300]}...")

            query_results.append(query_result)
            gap_inputs.append({
                "query_id": query_id,
                "query_text": query_text,
                "highest_score": highest_score,
                "fallback_triggered": confidence["fallback"],
            })

        # ── Gap detection ──────────────────────────────────────────
        state("GAP_DETECTION")
        s10_gap.detect(gap_inputs)

        # ── Audit export ───────────────────────────────────────────
        state("AUDIT_EXPORTED")
        s11_audit.export(query_results)

        print("\n" + "="*60)
        print("  PIPELINE COMPLETE")
        print("="*60)

    except Exception:
        print("\n[PIPELINE ERROR]")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run()
