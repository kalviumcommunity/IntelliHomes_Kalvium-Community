from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipeline.runner import run_pipeline

load_dotenv()

app = FastAPI(title="IntelliHomes RAG API")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    k: int = Field(
        3, ge=1, le=20, description="Number of retrieval chunks to return"
    )


class SourceItem(BaseModel):
    id: Optional[str]
    source: Optional[str]
    section: Optional[str]
    position: Optional[str]
    category: Optional[str]
    score: Optional[float]
    text: Optional[str]


class QueryResponse(BaseModel):
    status: str
    query: str
    answer: str
    sources: List[SourceItem]
    timings_ms: Dict[str, float]
    metadata: Dict[str, Any] = {}


@app.get("/")
def root():
    try:
        return {"message": "Welcome to IntelliHomes backend"}

    except Exception as exc:
        raise HTTPException(
            status_code=503, detail={"status": "not_ready", "error": str(exc)}
        )


@app.get("/health")
def health():
    try:
        return {"status": "healthy"}

    except Exception as exc:
        raise HTTPException(
            status_code=503, detail={"status": "not_ready", "error": str(exc)}
        )


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """Accept a question, run the RAG pipeline, and return a structured JSON response.

    Environment configuration (loaded from `.env` or environment):
    - OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
    - CHROMA_PATH, COLLECTION_NAME
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(
            status_code=400, detail="`question` must be a non-empty string"
        )

    try:
        # Run the pipeline. The pipeline reads model/DB config from env vars.
        result = run_pipeline(question, k=req.k)

        answer = result.get("answer", {}).get("answer") or ""
        sources = result.get("context", {}).get("sources", [])
        timings = result.get("timings_ms", {})

        return QueryResponse(
            status="ok",
            query=result.get("query", question),
            answer=answer,
            sources=sources,
            timings_ms=timings,
            metadata={
                "retrieval": result.get("retrieval"),
                "context_tokens": result.get("context", {}).get("context_tokens"),
            },
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:  # pragma: no cover - bubble up server errors
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {exc}\n{tb}"
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("API_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
