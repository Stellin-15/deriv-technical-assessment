"""
Multi-turn interactive CLI.
Run: python cli.py
Requires pipeline.py to have been run first (vectors.json must exist).
"""
import os
import sys

from stages import s4_retrieve, s5_confidence, s6_answer, s7_verify, s8_regenerate


def run_query(query: str, query_id: str, conversation_history: list) -> str:
    # Expand query with previous turn if history exists.
    if conversation_history:
        prev = conversation_history[-1]
        expanded_query = (
            f"Previous question: {prev['query']}\n"
            f"Previous answer summary: {prev['answer'][:300]}\n\n"
            f"Current question: {query}"
        )
    else:
        expanded_query = query

    retrieved = s4_retrieve.retrieve(expanded_query, query_id)
    confidence = s5_confidence.check(retrieved)

    if confidence["fallback"]:
        return (
            f"I was unable to find a confident answer in the Deriv Help Centre.\n"
            f"Reason: {confidence['reason']}\n"
            f"Best matching source: {confidence.get('best_source_url', 'N/A')}\n"
            f"Similarity score: {confidence.get('similarity_score', 0.0):.4f}\n"
            f"{confidence.get('suggested_action', '')}"
        )

    answer = s6_answer.generate(query, query_id, retrieved)
    claims = s7_verify.verify(answer, query_id, retrieved)
    final_answer, _ = s8_regenerate.regenerate(query, query_id, answer, retrieved, claims)
    return final_answer


def main():
    if not os.path.exists("vectors.json"):
        print("ERROR: vectors.json not found. Run 'python pipeline.py' first.")
        sys.exit(1)

    print("=" * 60)
    print("  Deriv Help Centre Assistant")
    print("=" * 60)
    print("Type your question or 'quit' to exit")
    print("Embeddings must be pre-computed. Run pipeline.py first.")
    print()

    conversation_history = []
    turn = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("q", "quit", "exit"):
            print("Goodbye.")
            break

        turn += 1
        query_id = f"CLI_{turn}"

        print()
        answer = run_query(user_input, query_id, conversation_history)
        print(f"Assistant: {answer}")
        print()

        conversation_history.append({"query": user_input, "answer": answer})


if __name__ == "__main__":
    main()
