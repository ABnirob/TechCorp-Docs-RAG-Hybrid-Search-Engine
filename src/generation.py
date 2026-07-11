"""
Answer generation from retrieved chunks.

Two backends, selected automatically:
  1. Anthropic API (if ANTHROPIC_API_KEY is set) -- a real LLM writes a
     grounded answer citing chunk IDs.
  2. Offline extractive fallback (no API key needed) -- returns the most
     relevant sentences from the top chunks with citations. This keeps the
     whole project runnable end-to-end with zero setup and zero cost,
     which matters for a portfolio piece a recruiter might actually try
     to run.

The prompt enforces citation of chunk IDs and explicitly instructs the
model to say when it doesn't know, which is the main lever against
hallucination in a RAG system -- it's not just "give the model context",
it's "constrain what the model is allowed to claim".
"""

from __future__ import annotations

from src import config
from src.retrieval import RetrievedChunk

SYSTEM_PROMPT = """You are a support assistant for TechCorp Cloud. Answer the user's \
question using ONLY the provided context passages. Rules:
1. Cite the chunk_id(s) you used in square brackets, e.g. [auth-001-0], after each claim.
2. If the context does not contain the answer, say clearly: "I don't have enough \
information in the knowledge base to answer that." Do not guess.
3. Be concise and direct. Do not repeat the question back.
"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for rc in chunks:
        c = rc.chunk
        blocks.append(f"[{c.chunk_id}] ({c.category} - {c.title})\n{c.text}")
    return "\n\n".join(blocks)


def generate_with_anthropic(query: str, retrieved: list[RetrievedChunk]) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    context = _format_context(retrieved)
    user_message = f"Context passages:\n\n{context}\n\nQuestion: {query}"

    response = client.messages.create(
        model=config.GENERATION_MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def generate_extractive(query: str, retrieved: list[RetrievedChunk]) -> str:
    """
    Offline fallback used when no API key is configured. Not an LLM --
    just returns the top passages verbatim with citations, so the pipeline
    is still runnable and testable without any external dependency.
    """
    if not retrieved:
        return "I don't have enough information in the knowledge base to answer that."

    lines = ["Here is what the knowledge base says (extractive mode -- no LLM key configured):\n"]
    for rc in retrieved:
        c = rc.chunk
        lines.append(f"- {c.text} [{c.chunk_id}]")
    return "\n".join(lines)


def generate_answer(query: str, retrieved: list[RetrievedChunk]) -> dict:
    """Returns {"answer": str, "backend": str} so callers/eval can tell which path ran."""
    if config.ANTHROPIC_API_KEY:
        try:
            answer = generate_with_anthropic(query, retrieved)
            return {"answer": answer, "backend": "anthropic"}
        except Exception as e:
            # Fail soft: still return a usable answer instead of crashing the demo.
            fallback = generate_extractive(query, retrieved)
            return {"answer": f"[LLM generation failed: {e}]\n\n{fallback}", "backend": "extractive-fallback-error"}
    else:
        answer = generate_extractive(query, retrieved)
        return {"answer": answer, "backend": "extractive"}
