"""
Answer-quality evaluation: is the generated answer actually grounded in the
retrieved context, or did the model (or extractive fallback) drift/hallucinate?

Two modes:
  1. LLM-as-judge (if ANTHROPIC_API_KEY is set): asks Claude to score the
     answer's faithfulness to the provided context on a 1-5 scale and flag
     any unsupported claims. This is the gold-standard approach used in
     production RAG evals (see RAGAS, TruLens, etc. for prior art).
  2. Heuristic fallback (no API key needed): a lightweight lexical-overlap
     check -- what fraction of the answer's content words are found
     somewhere in the retrieved context. Crude, but it is a real, cheap,
     zero-cost signal that catches gross hallucination (e.g. the model
     inventing a number that appears nowhere in context) and lets the eval
     harness run in CI without hitting a paid API on every commit.

Run: python -m evaluation.evaluate_answers
"""

from __future__ import annotations

import json
import re

from src import config
from src.rag_pipeline import RagPipeline

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "in", "on",
    "for", "and", "or", "with", "at", "by", "from", "that", "this", "it", "as", "your",
    "you", "i", "does", "do", "can", "will", "if", "not", "no", "what", "how", "when",
    "where", "which", "who", "my", "me", "have", "has", "had",
}


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def heuristic_faithfulness(answer: str, context_text: str) -> float:
    """Fraction of the answer's content words that also appear in the retrieved context."""
    answer_words = _content_words(answer)
    if not answer_words:
        return 0.0
    context_words = _content_words(context_text)
    grounded = answer_words & context_words
    return round(len(grounded) / len(answer_words), 4)


def llm_judge_faithfulness(query: str, answer: str, context_text: str) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    judge_prompt = f"""You are evaluating a RAG system's answer for faithfulness to its retrieved context.

Question: {query}

Retrieved context:
{context_text}

Generated answer:
{answer}

Score the answer's faithfulness from 1-5:
5 = fully supported by context, no invented facts
3 = mostly supported, minor unsupported additions
1 = largely unsupported or contradicts the context

Respond ONLY with valid JSON: {{"score": <int 1-5>, "unsupported_claims": ["..."], "reasoning": "one sentence"}}"""

    response = client.messages.create(
        model=config.GENERATION_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": None, "unsupported_claims": [], "reasoning": f"Could not parse judge output: {raw[:200]}"}


def main():
    with open(config.EVAL_QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    pipeline = RagPipeline()
    use_llm_judge = bool(config.ANTHROPIC_API_KEY)
    print(f"Judging mode: {'LLM-as-judge (Anthropic API)' if use_llm_judge else 'heuristic lexical-overlap (no API key set)'}\n")

    results = []
    for item in queries:
        query = item["query"]
        out = pipeline.query(query)
        context_text = "\n".join(c["text"] for c in out["retrieved_chunks"])

        if use_llm_judge and out["generation_backend"] == "anthropic":
            judged = llm_judge_faithfulness(query, out["answer"], context_text)
            score = judged.get("score")
        else:
            score = heuristic_faithfulness(out["answer"], context_text)
            judged = {"score": score, "unsupported_claims": [], "reasoning": "heuristic lexical overlap ratio"}

        results.append({
            "query": query,
            "generation_backend": out["generation_backend"],
            "faithfulness_score": score,
            "reasoning": judged.get("reasoning", ""),
        })

    scored = [r["faithfulness_score"] for r in results if isinstance(r["faithfulness_score"], (int, float))]
    avg = round(sum(scored) / len(scored), 4) if scored else None

    print(f"Evaluated {len(results)} queries. Average faithfulness score: {avg}\n")
    for r in results[:5]:
        print(f"  [{r['faithfulness_score']}] {r['query'][:60]}")
    if len(results) > 5:
        print(f"  ... and {len(results) - 5} more (see full results file)")

    with open(config.DATA_DIR / "answer_eval_results.json", "w") as f:
        json.dump({"average_faithfulness": avg, "results": results}, f, indent=2)
    print(f"\nSaved detailed results to {config.DATA_DIR / 'answer_eval_results.json'}")


if __name__ == "__main__":
    main()
