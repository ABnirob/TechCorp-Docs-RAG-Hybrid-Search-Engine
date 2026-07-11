"""
Builds the sparse (BM25) and dense (TF-IDF or sentence-embedding) indices
used by the hybrid retriever.

Design note: we default to TF-IDF cosine similarity instead of neural
embeddings so the whole project builds and runs fully offline with no
model download and no GPU -- important for CI, restricted networks, and
anyone who just cloned this repo and wants it to work in 10 seconds.
Setting USE_SENTENCE_EMBEDDINGS=true in .env swaps in real
sentence-transformer embeddings with no other code changes, since both
paths implement the same `DenseIndex` interface.
"""

from __future__ import annotations

import json
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunking import Chunk, chunk_documents
from src import config


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class SparseIndex:
    """BM25 keyword index -- strong on exact terms, IDs, error codes, acronyms."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in ranked]


class DenseIndex:
    """
    Dense semantic index. Two interchangeable backends:
      - TfidfBackend: no downloads, pure scikit-learn, works everywhere.
      - SentenceTransformerBackend: real embeddings, better recall on
        paraphrased/synonym-heavy queries, requires `sentence-transformers`
        and a one-time model download.
    """

    def __init__(self, chunks: list[Chunk], use_sentence_embeddings: bool = False):
        self.chunks = chunks
        self.use_sentence_embeddings = use_sentence_embeddings
        texts = [c.text for c in chunks]

        if use_sentence_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "USE_SENTENCE_EMBEDDINGS=true but sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                ) from e
            self.model = SentenceTransformer(config.SENTENCE_EMBEDDING_MODEL)
            self.doc_vectors = self.model.encode(texts, normalize_embeddings=True)
        else:
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.doc_vectors = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if self.use_sentence_embeddings:
            query_vec = self.model.encode([query], normalize_embeddings=True)
            scores = cosine_similarity(query_vec, self.doc_vectors)[0]
        else:
            query_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.doc_vectors)[0]

        ranked = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in ranked]


def load_documents(path=None) -> list[dict]:
    path = path or config.DOCS_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_indices(
    docs: list[dict] | None = None,
    strategy: str = None,
    chunk_size_words: int = None,
    overlap_words: int = None,
    use_sentence_embeddings: bool = None,
) -> tuple[list[Chunk], SparseIndex, DenseIndex]:
    docs = docs if docs is not None else load_documents()
    strategy = strategy or config.CHUNK_STRATEGY
    chunk_size_words = chunk_size_words or config.CHUNK_SIZE_WORDS
    overlap_words = overlap_words if overlap_words is not None else config.CHUNK_OVERLAP_WORDS
    use_sentence_embeddings = (
        use_sentence_embeddings if use_sentence_embeddings is not None else config.USE_SENTENCE_EMBEDDINGS
    )

    chunks = chunk_documents(docs, strategy, chunk_size_words, overlap_words)
    sparse_index = SparseIndex(chunks)
    dense_index = DenseIndex(chunks, use_sentence_embeddings=use_sentence_embeddings)
    return chunks, sparse_index, dense_index
