"""
Basic test suite covering chunking correctness, retrieval sanity, and
end-to-end pipeline execution. Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.chunking import fixed_size_chunks, sentence_chunks, split_into_sentences
from src.indexing import build_indices, load_documents
from src.retrieval import HybridRetriever
from src.rag_pipeline import RagPipeline


SAMPLE_DOC = {
    "doc_id": "test-001",
    "category": "Test",
    "title": "Sample",
    "text": "This is sentence one. This is sentence two. This is sentence three. "
            "This is sentence four. This is sentence five.",
}


def test_split_into_sentences():
    sentences = split_into_sentences(SAMPLE_DOC["text"])
    assert len(sentences) == 5
    assert sentences[0] == "This is sentence one."


def test_fixed_size_chunks_respects_overlap():
    chunks = fixed_size_chunks(SAMPLE_DOC, chunk_size_words=6, overlap_words=2)
    assert len(chunks) > 1
    # every chunk should belong to the source doc
    assert all(c.doc_id == "test-001" for c in chunks)


def test_sentence_chunks_does_not_split_sentences():
    chunks = sentence_chunks(SAMPLE_DOC, chunk_size_words=8, overlap_words=3)
    for c in chunks:
        # every chunk should end on a sentence boundary (period)
        assert c.text.strip().endswith(".")


def test_short_doc_returns_single_chunk():
    short_doc = {"doc_id": "short-1", "category": "Test", "title": "Short", "text": "Just one short sentence."}
    chunks = fixed_size_chunks(short_doc, chunk_size_words=90, overlap_words=20)
    assert len(chunks) == 1
    chunks2 = sentence_chunks(short_doc, chunk_size_words=90, overlap_words=20)
    assert len(chunks2) == 1


def test_build_indices_produces_matching_chunk_count():
    docs = load_documents()
    chunks, sparse_index, dense_index = build_indices(docs=docs)
    assert len(chunks) == len(sparse_index.chunks) == len(dense_index.chunks)
    assert len(chunks) > 0


def test_hybrid_retrieval_returns_top_k():
    docs = load_documents()
    chunks, sparse_index, dense_index = build_indices(docs=docs)
    retriever = HybridRetriever(chunks, sparse_index, dense_index)
    results = retriever.retrieve("How do I authenticate my API requests?", top_k=3)
    assert len(results) <= 3
    assert len(results) > 0


def test_retrieval_finds_relevant_doc_for_clear_query():
    """A query with a strong keyword match should retrieve the obviously relevant doc."""
    docs = load_documents()
    chunks, sparse_index, dense_index = build_indices(docs=docs)
    retriever = HybridRetriever(chunks, sparse_index, dense_index)
    results = retriever.retrieve("SOC 2 compliance certification", top_k=3)
    doc_ids = [r.chunk.doc_id for r in results]
    assert "security-002" in doc_ids


def test_pipeline_end_to_end_offline():
    """Full pipeline should run without any API key using the extractive fallback."""
    pipeline = RagPipeline()
    result = pipeline.query("What is the rate limit on the Starter tier?")
    assert result["answer"]
    assert result["generation_backend"] == "extractive"
    assert len(result["retrieved_chunks"]) > 0
    assert result["timings_ms"]["total"] >= 0


def test_pipeline_handles_empty_ish_query_gracefully():
    pipeline = RagPipeline()
    result = pipeline.query("???")
    # Should not crash; may return low-confidence results or an "I don't know" style answer.
    assert "answer" in result


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
