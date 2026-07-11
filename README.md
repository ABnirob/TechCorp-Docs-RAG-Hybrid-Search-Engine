# TechCorp Docs RAG — Hybrid Search Engine

# <span style="color:blue">TechCorp Docs RAG — Production-Grade Hybrid Search Engine</span>

[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

A production-shaped Retrieval-Augmented Generation (RAG) system built deliberately to demonstrate the rigorous engineering decisions that separate a transient "notebook demo" from a resilient, enterprise-ready search architecture. This project showcases chunking strategy trade-offs, hybrid sparse+dense retrieval mechanics, automated regression evaluation with quantitative metrics, and containerized API deployment.

The system executes over a synthetic enterprise knowledge base ("TechCorp Cloud") comprising 28 comprehensive source documents spanning critical operational domains: Authentication, Billing, Rate Limits, Automated Deployments, Observability, Infrastructure Security, SDK Lifecycles, Webhooks, and Root-Cause Troubleshooting.

> **Zero-Dependency Core:** Runs fully offline out of the box with zero third-party API keys, local model weight downloads, or hardware acceleration required. Simply plug in an `ANTHROPIC_API_KEY` to instantly upgrade from local extractive fallbacks to an enterprise-grade, deterministic synthesis loop with verifiable inline citations.

---

## <span style="color:blue">Why This Project Exists</span>

Standard RAG portfolio projects routinely collapse under production edge cases because they stop at basic vector abstractions over static documents. This repository is architected explicitly around the non-trivial questions faced during system design reviews for mid-to-senior AI Platform Engineering roles:

* **Verifiable Metrics Over Intuition:** Instead of assuming chunking configurations are optimal, this architecture employs an isolated evaluation harness providing quantitative regression tracking across standardized information retrieval metrics.
* **Retriever Complementarity:** Rather than relying exclusively on dense vector spaces—which notoriously degrade on precise alphanumeric identifiers, serial numbers, and system error codes—this engine pairs a lexical token index (BM25) with semantic representations to construct a resilient, hybrid retrieval boundary.
* **Determinism & Verification:** To mitigate semantic hallucinations, the generation layer exposes an isolated faithfulness validation loop utilizing both an LLM-as-a-Judge protocol and deterministic, zero-cost fallback heuristics.
* **Production-Grade Infrastructure:** The core runtime is exposed through a multi-worker FastAPI application complete with structural data schema validation, comprehensive unit/integration test assertions, performance metrics endpoints, and an optimized, multi-stage Docker execution environment.

---

## <span style="color:blue">Architecture Topology</span>

<img width="2720" height="2360" alt="rag_hybrid_pipeline_architecture" src="https://github.com/user-attachments/assets/9fc043b8-c71e-48bc-b5cd-b77df84ce55c" />


 

**Runs fully offline out of the box — no API key, no GPU, no model download required.** Plug in an `ANTHROPIC_API_KEY` to swap the extractive fallback for real LLM-generated, cited answers.

---

## Why this project

Most RAG portfolio projects stop at "I built a chatbot over my PDFs." This one is built around the questions an interviewer actually asks in a mid-level AI Engineer loop:

- *How do you know your chunking strategy is any good?* → There's an eval harness that measures it (see [Evaluation Results](#evaluation-results)).
- *Why hybrid search instead of just embeddings?* → Sparse (BM25) and dense retrieval fail on different query types; the eval quantifies where each wins.
- *How do you know the LLM isn't hallucinating?* → There's a faithfulness/groundedness check, with an LLM-as-judge path and an offline heuristic fallback.
- *Can this actually run in production?* → It's behind a FastAPI service, has tests, a health check, and a Dockerfile.

---

## Architecture

```
                    ┌─────────────────────┐
                    │  techcorp_docs.json │   (28 source documents)
                    └──────────┬──────────┘
                               │
                       ┌───────▼────────┐
                       │    Chunking     │  sentence-aware, 90 words/chunk,
                       │  (chunking.py)  │  20-word overlap  →  29 chunks
                       └───────┬────────┘
                               │
                ┌──────────────┼───────────────┐
                │                              │
       ┌────────▼────────┐           ┌─────────▼─────────┐
       │   Sparse Index   │           │    Dense Index     │
       │   BM25Okapi       │           │  TF-IDF / Sentence │
       │  (indexing.py)    │           │  Transformers      │
       └────────┬────────┘           └─────────┬─────────┘
                │                              │
                └──────────────┬───────────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Hybrid Retriever      │  Reciprocal Rank Fusion
                    │  (retrieval.py)        │  (or weighted score fusion)
                    └──────────┬───────────┘
                               │  top-k chunks
                    ┌──────────▼───────────┐
                    │  Answer Generation     │  Anthropic API (cited answer)
                    │  (generation.py)       │  or offline extractive fallback
                    └──────────┬───────────┘
                               │
                ┌──────────────┼───────────────┐
       ┌────────▼────────┐           ┌─────────▼─────────┐
       │   FastAPI (/query)│           │  CLI demo           │
       │   api/app.py       │           │  scripts/demo_cli.py│
       └────────────────────┘           └─────────────────────┘
```

All four entry points (CLI, API, retrieval eval, answer eval) call the same `RagPipeline` class in `src/rag_pipeline.py`, so there's exactly one code path to reason about and test.

---

## Quickstart

```bash
git clone <this-repo>
cd rag-hybrid-search-engine
pip install -r requirements.txt

# Try it immediately, no setup required:
python scripts/demo_cli.py
```

To use real LLM generation instead of the offline extractive fallback:

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
python scripts/demo_cli.py
```

Run the API server:

```bash
uvicorn api.app:app --reload --port 8000
# then in another terminal:
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What happens if I exceed my rate limit?"}'
```

Or with Docker:

```bash
docker build -t techcorp-rag .
docker run -p 8000:8000 --env-file .env techcorp-rag
```

Run the test suite:

```bash
pytest tests/ -v
```

---

## Evaluation

### Retrieval quality (`python -m evaluation.evaluate_retrieval`)

Compares three retrieval configurations against 25 hand-labeled queries with known-relevant documents:

| Mode   | Precision@5 | Recall@5 | MRR    | Hit Rate@5 |
|--------|------------|----------|--------|------------|
| Sparse (BM25 only)  | 0.192 | 0.880 | 0.823 | 0.92 |
| Dense (TF-IDF only)  | 0.208 | 0.940 | 0.913 | 0.96 |
| Hybrid (RRF fusion)  | 0.208 | 0.940 | 0.887 | 0.96 |

**Honest read of this result, not the one that makes the demo look best:** on this small, single-domain corpus, hybrid fusion doesn't clearly beat dense-only retrieval, and the TF-IDF dense backend already edges out BM25 on its own. That's expected here, not a bug — TF-IDF is itself a lexical (word-overlap) method with bigram support, so it captures much of the same signal BM25 does, which shrinks the complementary gain that hybrid search is supposed to unlock. Hybrid search earns its keep most clearly when the dense side is a *true semantic* embedding (paraphrases, synonyms) that BM25 structurally cannot match, and on larger, more heterogeneous corpora where sparse and dense retrievers disagree more often. Set `USE_SENTENCE_EMBEDDINGS=true` in `.env` (requires `pip install sentence-transformers`) and re-run the eval to see this gap open up — the retrieval and fusion code is identical either way, only the dense backend changes, by design (see `DenseIndex` in `src/indexing.py`).

This is exactly the kind of nuance worth being able to explain in an interview: hybrid search is not a free lunch, it's a hedge against the failure modes of whichever single retriever you'd otherwise rely on, and its value is corpus- and query-distribution-dependent — which is why you eval it instead of assuming it.

### Answer faithfulness (`python -m evaluation.evaluate_answers`)

Checks whether generated answers are actually grounded in the retrieved context rather than invented:

- **With `ANTHROPIC_API_KEY` set:** an LLM-as-judge scores each answer 1-5 for faithfulness and flags unsupported claims.
- **Without a key (default):** a lexical-overlap heuristic measures what fraction of the answer's content words appear in the retrieved context — a crude but zero-cost sanity check that still catches gross hallucination, and keeps this eval runnable in CI with no API cost.

Both eval scripts write full per-query results to `data/*_eval_results.json` for further analysis.

---

## Design decisions & tradeoffs

**Chunking — sentence-aware over fixed-size.** Fixed-size windows are simpler but can slice a sentence in half, which hurts both BM25 (breaks term proximity) and embeddings (chunk no longer represents one coherent idea). `chunking.py` implements both strategies behind the same interface (`CHUNK_STRATEGY=fixed|sentence` in `.env`) specifically so this is a testable decision, not an assumption.

**TF-IDF as the default dense backend, not sentence-transformers.** This was a deliberate choice to keep the project runnable in 10 seconds with no model download, no GPU, and no network dependency — important for anyone actually cloning this to try it, and for CI. The `DenseIndex` class supports both backends behind one interface, so swapping in real embeddings is a one-line config change, not a rewrite.

**Reciprocal Rank Fusion over weighted score averaging.** BM25 and cosine-similarity scores live on incomparable scales, so naively weighting and summing them is fragile — a corpus-size change alone can shift BM25's score range. RRF fuses by *rank* instead of raw score, which is scale-free and is the standard approach in production hybrid search systems. Weighted fusion is still implemented (`FUSION_STRATEGY=weighted`) for comparison.

**Extractive fallback instead of a required API key.** A portfolio project a recruiter can't actually run is a worse portfolio project. The generation layer fails soft: no key → return the top passages with citations; API call fails for any reason → catch it and fall back rather than crash the demo.

**What I'd do differently with more time:**
- Add a re-ranker (e.g., a cross-encoder) as a second-stage over the top-20 candidates from hybrid retrieval — this consistently gives the largest single quality bump in production RAG systems and is the natural next experiment this eval harness is set up to support.
- Grow the eval set past 25 queries and add harder negative examples (plausible-but-wrong documents) to stress-test precision, not just recall.
- Add response caching for repeated queries and a persisted index (currently rebuilt in memory on startup, which is fine at this corpus size but wouldn't scale to a real document set).

---

## Project structure

```
rag-hybrid-search-engine/
├── data/
│   ├── techcorp_docs.json        # 28 synthetic source documents
│   └── eval_queries.json         # 25 labeled queries w/ ground-truth relevant doc_ids
├── src/
│   ├── config.py                 # all tunable settings in one place
│   ├── chunking.py                # fixed-size + sentence-aware chunking strategies
│   ├── indexing.py                # BM25 + TF-IDF/sentence-embedding indices
│   ├── retrieval.py               # hybrid retriever, RRF + weighted fusion
│   ├── generation.py              # Anthropic API + offline extractive fallback
│   └── rag_pipeline.py            # single orchestration entry point
├── evaluation/
│   ├── evaluate_retrieval.py      # Precision@k, Recall@k, MRR, Hit Rate@k
│   └── evaluate_answers.py        # faithfulness / groundedness scoring
├── api/
│   └── app.py                     # FastAPI service (/query, /healthz)
├── scripts/
│   ├── demo_cli.py                # interactive terminal demo
│   └── build_index.py             # inspect chunking/index output
├── tests/
│   └── test_pipeline.py           # 9 tests covering chunking, retrieval, e2e pipeline
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Tech stack

Python 3.12 · FastAPI · rank_bm25 · scikit-learn (TF-IDF) · optional sentence-transformers · Anthropic API · pytest · Docker

## License

MIT — see [LICENSE](LICENSE).
