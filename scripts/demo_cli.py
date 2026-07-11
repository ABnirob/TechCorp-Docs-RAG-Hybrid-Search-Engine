"""
Interactive terminal demo. Run with: python scripts/demo_cli.py

Type a question about "TechCorp Cloud" (the fictional product the demo
knowledge base documents) and see the retrieved chunks, fused scores, and
generated answer. Type 'quit' to exit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_pipeline import RagPipeline
from src import config


def print_divider():
    print("-" * 70)


def main():
    print("Building indices...")
    pipeline = RagPipeline()
    print(f"Indexed {len(pipeline.chunks)} chunks from the TechCorp Cloud docs.")
    backend = "Anthropic API" if config.ANTHROPIC_API_KEY else "offline extractive fallback (set ANTHROPIC_API_KEY for real generation)"
    print(f"Generation backend: {backend}")
    print_divider()
    print("Try questions like:")
    print("  - What happens if I exceed my rate limit?")
    print("  - How do I roll back a bad deployment?")
    print("  - Is TechCorp SOC 2 compliant?")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            question = input("Ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            break

        result = pipeline.query(question)

        print_divider()
        print("ANSWER:")
        print(result["answer"])
        print()
        print(f"Retrieved chunks (fused score, sparse rank, dense rank):")
        for c in result["retrieved_chunks"]:
            print(f"  [{c['chunk_id']}] score={c['fused_score']} sparse_rank={c['sparse_rank']} dense_rank={c['dense_rank']} -- {c['title']}")
        print(f"\nTiming: retrieval={result['timings_ms']['retrieval']}ms | "
              f"generation={result['timings_ms']['generation']}ms | "
              f"total={result['timings_ms']['total']}ms")
        print_divider()
        print()


if __name__ == "__main__":
    main()
