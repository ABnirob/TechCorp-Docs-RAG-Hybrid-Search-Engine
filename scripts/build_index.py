"""
Standalone script to build the index and print a summary -- useful for
sanity-checking chunking output before running the full pipeline, and as
a template for pre-warming a persisted index in a real deployment.

Run: python scripts/build_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indexing import build_indices, load_documents
from src import config


def main():
    docs = load_documents()
    chunks, sparse_index, dense_index = build_indices(docs=docs)

    print(f"Loaded {len(docs)} source documents.")
    print(f"Chunking strategy: {config.CHUNK_STRATEGY} "
          f"(size={config.CHUNK_SIZE_WORDS} words, overlap={config.CHUNK_OVERLAP_WORDS} words)")
    print(f"Produced {len(chunks)} chunks.\n")

    print("Sample chunks:")
    for c in chunks[:5]:
        word_count = len(c.text.split())
        print(f"  [{c.chunk_id}] ({word_count} words) {c.title}: {c.text[:100]}...")

    avg_words = sum(len(c.text.split()) for c in chunks) / len(chunks)
    print(f"\nAverage chunk size: {avg_words:.1f} words")
    print(f"Dense index backend: {'sentence-transformers' if dense_index.use_sentence_embeddings else 'TF-IDF'}")


if __name__ == "__main__":
    main()
