"""
Chunking strategies for splitting source documents into retrievable passages.

Why this matters (and why it's not a solved problem):
- Chunks too large -> retrieval gets noisy, embeddings blur multiple topics
  together, and you waste context-window budget on irrelevant text.
- Chunks too small -> you lose surrounding context needed to answer the
  question, and you multiply the number of near-duplicate vectors in the index.
- Splitting mid-sentence breaks both keyword search (BM25 term proximity)
  and dense embeddings (the chunk no longer represents a coherent idea).

This module implements two strategies so they can be A/B tested via the
eval harness (see evaluation/evaluate_retrieval.py):

1. fixed_size_chunks   - naive rolling window over words, with overlap.
   Fast, simple, works on any text, but can cut sentences in half.
2. sentence_chunks     - respects sentence boundaries, packing sentences
   into a chunk until it would exceed the target size. This is what we
   use by default, since it consistently scored better on our retrieval
   eval (see evaluation/README notes) at a negligible performance cost.
"""

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    category: str
    title: str
    text: str


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str) -> list[str]:
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s for s in sentences if s]


def fixed_size_chunks(doc: dict, chunk_size_words: int, overlap_words: int) -> list[Chunk]:
    words = doc["text"].split()
    if len(words) <= chunk_size_words:
        return [Chunk(f"{doc['doc_id']}-0", doc["doc_id"], doc["category"], doc["title"], doc["text"])]

    chunks = []
    start = 0
    idx = 0
    step = max(chunk_size_words - overlap_words, 1)
    while start < len(words):
        window = words[start : start + chunk_size_words]
        chunk_text = " ".join(window)
        chunks.append(Chunk(f"{doc['doc_id']}-{idx}", doc["doc_id"], doc["category"], doc["title"], chunk_text))
        idx += 1
        start += step
    return chunks


def sentence_chunks(doc: dict, chunk_size_words: int, overlap_words: int) -> list[Chunk]:
    sentences = split_into_sentences(doc["text"])
    if not sentences:
        return []

    chunks = []
    current: list[str] = []
    current_word_count = 0
    idx = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        if current and current_word_count + sentence_word_count > chunk_size_words:
            chunk_text = " ".join(current)
            chunks.append(Chunk(f"{doc['doc_id']}-{idx}", doc["doc_id"], doc["category"], doc["title"], chunk_text))
            idx += 1
            # carry the last sentence(s) forward as overlap for continuity
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current):
                w = len(s.split())
                if overlap_count + w > overlap_words:
                    break
                overlap_sentences.insert(0, s)
                overlap_count += w
            current = overlap_sentences
            current_word_count = overlap_count

        current.append(sentence)
        current_word_count += sentence_word_count

    if current:
        chunk_text = " ".join(current)
        chunks.append(Chunk(f"{doc['doc_id']}-{idx}", doc["doc_id"], doc["category"], doc["title"], chunk_text))

    return chunks


def chunk_documents(docs: list[dict], strategy: str, chunk_size_words: int, overlap_words: int) -> list[Chunk]:
    """Chunk a list of source documents using the given strategy."""
    fn = {"fixed": fixed_size_chunks, "sentence": sentence_chunks}.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown chunk strategy '{strategy}'. Use 'fixed' or 'sentence'.")

    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(fn(doc, chunk_size_words, overlap_words))
    return all_chunks
