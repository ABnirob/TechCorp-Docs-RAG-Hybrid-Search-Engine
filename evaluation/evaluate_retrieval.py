"""
Retrieval evaluation harness.

Loads the labeled query set (data/eval_queries.json, each query mapped to
the ground-truth relevant doc_ids) and computes standard IR metrics:

  - Precision@k : of the top-k chunks returned, what fraction are relevant?
  - Recall@k    : of all relevant docs, what fraction did we retrieve in top-k?
  - MRR         : how high up was the FIRST relevant result, on average?
  - Hit Rate@k  : fraction of queries where at least one relevant doc appeared

We evaluate three configurations to make the hybrid-vs-single-retriever
argument with actual numbers instead of a claim:
  1. Sparse only (BM25)
  2. Dense only (TF-IDF / embeddings)
  3. Hybrid (RRF fusion of both)

Run: python -m evaluation.evaluate_retrieval
"""

from __future__ import annotations

import json
from src import config
from src.indexing import build_indices, load_documents
from src.retrieval import HybridRetriever


def load_eval_queries():
    with open(config.EVAL_QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def precision_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    top_k = retrieved_doc_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for d in top_k if d in relevant_doc_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    hits = sum(1 for d in top_k if d in relevant_doc_ids)
    return hits / len(relevant_doc_ids)


def reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def hit_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    return 1.0 if any(d in relevant_doc_ids for d in retrieved_doc_ids[:k]) else 0.0


def evaluate_retriever(retriever: HybridRetriever, mode: str, queries: list[dict], k: int = 5) -> dict:
    precisions, recalls, rrs, hits = [], [], [], []

    for item in queries:
        query = item["query"]
        relevant = set(item["relevant_doc_ids"])

        if mode == "sparse":
            retrieved = retriever.retrieve_sparse_only(query, top_k=k)
        elif mode == "dense":
            retrieved = retriever.retrieve_dense_only(query, top_k=k)
        elif mode == "hybrid":
            retrieved = retriever.retrieve(query, top_k=k, fusion_strategy="rrf")
        else:
            raise ValueError(mode)

        retrieved_doc_ids = [rc.chunk.doc_id for rc in retrieved]

        precisions.append(precision_at_k(retrieved_doc_ids, relevant, k))
        recalls.append(recall_at_k(retrieved_doc_ids, relevant, k))
        rrs.append(reciprocal_rank(retrieved_doc_ids, relevant))
        hits.append(hit_at_k(retrieved_doc_ids, relevant, k))

    n = len(queries)
    return {
        "mode": mode,
        "k": k,
        "num_queries": n,
        f"precision@{k}": round(sum(precisions) / n, 4),
        f"recall@{k}": round(sum(recalls) / n, 4),
        "mrr": round(sum(rrs) / n, 4),
        f"hit_rate@{k}": round(sum(hits) / n, 4),
    }


def main():
    docs = load_documents()
    queries = load_eval_queries()

    chunks, sparse_index, dense_index = build_indices(docs=docs)
    retriever = HybridRetriever(chunks, sparse_index, dense_index)

    print(f"Loaded {len(docs)} source docs -> {len(chunks)} chunks. Evaluating on {len(queries)} labeled queries.\n")

    results = []
    for mode in ["sparse", "dense", "hybrid"]:
        res = evaluate_retriever(retriever, mode, queries, k=config.TOP_K)
        results.append(res)

    header = f"{'Mode':<10}{'Precision@'+str(config.TOP_K):<16}{'Recall@'+str(config.TOP_K):<14}{'MRR':<10}{'Hit Rate@'+str(config.TOP_K):<12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['mode']:<10}"
            f"{r[f'precision@{config.TOP_K}']:<16}"
            f"{r[f'recall@{config.TOP_K}']:<14}"
            f"{r['mrr']:<10}"
            f"{r[f'hit_rate@{config.TOP_K}']:<12}"
        )

    with open(config.DATA_DIR / "retrieval_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {config.DATA_DIR / 'retrieval_eval_results.json'}")


if __name__ == "__main__":
    main()
