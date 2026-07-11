"""
FastAPI wrapper around the RAG pipeline.

Run locally:   uvicorn api.app:app --reload --port 8000
Docs UI:       http://localhost:8000/docs

This is intentionally small -- the point isn't to reproduce a full API
gateway, it's to show the pipeline behind a real HTTP interface with
request validation, structured responses, and a health check, which is
the minimum bar for "production-shaped" rather than "notebook-shaped".
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag_pipeline import RagPipeline

app = FastAPI(
    title="TechCorp Docs RAG API",
    description="Hybrid-search RAG over TechCorp Cloud's documentation.",
    version="1.0.0",
)

# Built once at startup and reused across requests -- rebuilding the index
# per-request would make every call pay chunking + BM25 + TF-IDF fit cost.
pipeline = RagPipeline()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The user's natural-language question")
    top_k: int | None = Field(None, ge=1, le=20, description="Number of chunks to retrieve")
    fusion_strategy: str | None = Field(None, description="'rrf' or 'weighted'")


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    fused_score: float
    sparse_rank: int | None
    dense_rank: int | None


class QueryResponse(BaseModel):
    query: str
    answer: str
    generation_backend: str
    retrieved_chunks: list[RetrievedChunkResponse]
    timings_ms: dict


@app.get("/healthz")
def healthz():
    return {"status": "ok", "chunks_indexed": len(pipeline.chunks)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if req.fusion_strategy and req.fusion_strategy not in ("rrf", "weighted"):
        raise HTTPException(status_code=400, detail="fusion_strategy must be 'rrf' or 'weighted'")

    result = pipeline.query(req.question, top_k=req.top_k, fusion_strategy=req.fusion_strategy)
    return result


@app.get("/")
def root():
    return {
        "message": "TechCorp Docs RAG API. POST /query with a JSON body {'question': '...'}. See /docs for the full schema.",
    }
