"""
Top-level orchestration: build indices once, then answer any number of
queries against them. This is the single entry point used by the CLI demo,
the FastAPI server, and the evaluation scripts, so all three are guaranteed
to exercise the exact same retrieval + generation logic.
"""

from __future__ import annotations

import time

from src.indexing import build_indices, load_documents
from src.retrieval import HybridRetriever
from src.generation import generate_answer


class RagPipeline:
    def __init__(self, docs: list[dict] | None = None, **index_kwargs):
        docs = docs if docs is not None else load_documents()
        self.chunks, self.sparse_index, self.dense_index = build_indices(docs=docs, **index_kwargs)
        self.retriever = HybridRetriever(self.chunks, self.sparse_index, self.dense_index)

    def query(self, question: str, top_k: int = None, fusion_strategy: str = None) -> dict:
        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(question, top_k=top_k, fusion_strategy=fusion_strategy)
        t1 = time.perf_counter()
        result = generate_answer(question, retrieved)
        t2 = time.perf_counter()

        return {
            "query": question,
            "answer": result["answer"],
            "generation_backend": result["backend"],
            "retrieved_chunks": [
                {
                    "chunk_id": rc.chunk.chunk_id,
                    "doc_id": rc.chunk.doc_id,
                    "title": rc.chunk.title,
                    "text": rc.chunk.text,
                    "fused_score": round(rc.score, 4),
                    "sparse_rank": rc.sparse_rank,
                    "dense_rank": rc.dense_rank,
                }
                for rc in retrieved
            ],
            "timings_ms": {
                "retrieval": round((t1 - t0) * 1000, 2),
                "generation": round((t2 - t1) * 1000, 2),
                "total": round((t2 - t0) * 1000, 2),
            },
        }
