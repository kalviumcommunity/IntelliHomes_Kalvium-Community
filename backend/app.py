from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.utils import secure_filename

from ingestion.loader import DocumentLoadError, UnsupportedFormatError, load_file
from prompts.renderer import render_prompt
from scripts.embed_corpus import (
    attach_vectors,
    chunk_corpus,
    embed_corpus_chunks,
    estimate_cost,
    estimate_tokens,
    save_store,
)
from scripts.index_corpus import index_store

app = Flask(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = BACKEND_ROOT / "uploads"
EMBEDDING_DIR = BACKEND_ROOT.parent / "data" / "embeddings"
MAX_UPLOAD_BYTES = int(os.environ.get("INTELLIHOMES_MAX_UPLOAD_BYTES", str(1_048_576)))
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".html",
    ".htm",
    ".md",
    ".markdown",
    ".txt",
    ".text",
}


@app.after_request
def add_cors_headers(response: Response) -> Response:
    response.headers.setdefault("Access-Control-Allow-Origin", "*")
    response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
    response.headers.setdefault("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok", "service": "IntelliHomes backend"})


def _safe_upload_path(filename: str) -> Path:
    safe_name = secure_filename(filename)
    if not safe_name:
        raise ValueError("file name is required")

    suffix = Path(safe_name).suffix.lower()
    stem = Path(safe_name).stem or "upload"
    unique = uuid.uuid4().hex[:8]
    target = UPLOAD_DIR / f"{stem}-{unique}{suffix}"
    return target


def _build_store_for_document(doc_path: Path, doc_source: str, *records: dict[str, Any]) -> dict[str, Any]:
    text_records = [r for r in records if r.get("text")]
    vectors = [r.get("vector") for r in text_records]
    valid_vectors = [v for v in vectors if v is not None]
    if not text_records or not valid_vectors:
        raise ValueError("uploaded document did not produce any valid chunk vectors")

    dim = len(valid_vectors[0])
    if any(len(v) != dim for v in valid_vectors):
        raise ValueError("embedding dimension mismatch across uploaded chunks")

    tokens = estimate_tokens([r["text"] for r in text_records])
    _, cost = estimate_cost(tokens, os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"))
    store = {
        "corpus": "upload",
        "documents": 1,
        "chunk_count": len(text_records),
        "total_chunks": len(text_records),
        "embedded": len(text_records),
        "skipped": 0,
        "failed": 0,
        "batches": 1,
        "batch_size": len(text_records),
        "dim": dim,
        "expected_dim": os.environ.get("EMBEDDING_DIM"),
        "model": os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"),
        "endpoint": os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        "mode": "live-or-simulated",
        "stats": {"retries": 0, "failed_batches": 0, "failed_starts": []},
        "tokens": tokens,
        "price_per_1m": 0.0,
        "cost_usd": round(cost, 6),
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "output_file": str(EMBEDDING_DIR / f"{Path(doc_source).stem}-{uuid.uuid4().hex[:8]}-embeddings.json"),
        "records": text_records,
    }
    return store


def _process_upload(file_path: Path) -> dict[str, Any]:
    doc = load_file(file_path)
    records = chunk_corpus([doc])
    if not records:
        raise ValueError("uploaded document produced no searchable chunks")

    texts = [record["text"] for record in records]
    mode, vectors, stats = embed_corpus_chunks(texts, use_live=os.environ.get("EMBEDDING_MODE", "").lower() != "simulated")
    for record, vector in zip(records, vectors):
        if vector is not None:
            record["vector"] = vector

    records = [record for record in records if record.get("vector") is not None]
    if not records:
        raise ValueError("uploaded document could not be embedded")

    store_path = EMBEDDING_DIR / f"{Path(doc.source).stem}-{uuid.uuid4().hex[:8]}-embeddings.json"
    store = {
        "corpus": Path(doc.source).stem,
        "documents": 1,
        "chunk_count": len(records),
        "total_chunks": len(records),
        "embedded": len(records),
        "skipped": 0,
        "failed": 0,
        "batches": 1,
        "batch_size": len(records),
        "dim": len(records[0]["vector"]),
        "expected_dim": os.environ.get("EMBEDDING_DIM"),
        "model": os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"),
        "endpoint": os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        "mode": mode,
        "stats": stats,
        "tokens": estimate_tokens(texts),
        "price_per_1m": 0.0,
        "cost_usd": round(estimate_cost(estimate_tokens(texts), os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"))[1], 6),
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "output_file": str(store_path),
        "records": records,
    }
    save_store(store, store_path)
    report = index_store(store_path=store_path, rebuild=False)

    return {
        "source": doc.source,
        "store": str(store_path),
        "chunks_indexed": report["indexed"],
        "collection_count": report["count"],
        "status": "uploaded",
    }


@app.post("/upload")
def upload_document() -> Any:
    file_obj = request.files.get("file")
    if file_obj is None or file_obj.filename == "":
        return jsonify({"status": "error", "message": "No file was uploaded."}), 400

    filename = secure_filename(file_obj.filename)
    if not filename:
        return jsonify({"status": "error", "message": "No file was uploaded."}), 400

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({"status": "error", "message": f"Unsupported file format: {suffix or 'unknown'}"}), 415

    bytes_data = file_obj.read()
    if not bytes_data or not bytes_data.strip():
        return jsonify({"status": "error", "message": "Uploaded file is empty."}), 400

    if len(bytes_data) > MAX_UPLOAD_BYTES:
        return jsonify({"status": "error", "message": f"File exceeds the {MAX_UPLOAD_BYTES} byte upload limit."}), 413

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_path = _safe_upload_path(filename)
    safe_path.write_bytes(bytes_data)

    try:
        result = _process_upload(safe_path)
        return jsonify({"status": "uploaded", "message": "Document uploaded and indexed successfully.", **result}), 201
    except UnsupportedFormatError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 415
    except (DocumentLoadError, ValueError, OSError) as exc:
        return jsonify({"status": "error", "message": f"Document processing failed: {exc}"}), 400
    except Exception as exc:  # pragma: no cover - defensive handler for runtime failures
        return jsonify({"status": "error", "message": f"Unexpected processing error: {exc}"}), 500


@app.post("/query")
def query_documents() -> Any:
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("query") or payload.get("question") or "").strip()
    if not question:
        return jsonify({"status": "error", "message": "A non-empty query is required."}), 400

    from retrieval.vector_search import search

    try:
        result = search(question, k=5)
    except Exception as exc:  # pragma: no cover - retrieval runtime guard
        return jsonify({"status": "error", "message": f"Query failed: {exc}"}), 500

    return jsonify({
        "status": "ok",
        "query": question,
        "results": [
            {
                "id": hit.get("id"),
                "score": hit.get("score"),
                "source": hit.get("metadata", {}).get("source"),
                "section": hit.get("metadata", {}).get("section"),
                "chunk_id": hit.get("metadata", {}).get("chunk_id") or hit.get("id"),
                "text": hit.get("text"),
            }
            for hit in result.get("results", [])
        ],
    })


def _citation_payload(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        citations.append(
            {
                "marker": f"[{index}]",
                "id": source.get("id"),
                "document": source.get("source") or "unknown",
                "section": source.get("section") or "unknown",
                "position": source.get("position"),
                "chunk_id": source.get("id"),
                "score": source.get("score"),
                "text": source.get("text", ""),
            }
        )
    return citations


def _iter_text_chunks(text: str, chunk_size: int = 24) -> Iterator[str]:
    if not text:
        return
    start = 0
    while start < len(text):
        yield text[start : start + chunk_size]
        start += chunk_size


def _stream_live_generation(question: str, context: str) -> Iterator[str]:
    from openai import OpenAI

    model = os.environ.get("OPENAI_MODEL", "llama3.1:8b")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured for live streaming")

    system_prompt, user_prompt = render_prompt(context=context, question=question)
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL"),
        timeout=60,
    )
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        stream=True,
    )
    for event in stream:
        delta = ""
        if event.choices and event.choices[0].delta:
            delta = event.choices[0].delta.content or ""
        if delta:
            yield delta


def _stream_chat_events(question: str, k: int) -> Iterator[str]:
    from pipeline.generate import generate_stage
    from pipeline.runner import run_pipeline

    def _event(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=True) + "\n"

    answer_buffer = ""
    citations: list[dict[str, Any]] = []

    try:
        yield _event({"type": "status", "message": "Retrieving grounded sources..."})
        result = run_pipeline(question, k=max(1, k), live=False)
        sources = result.get("context", {}).get("sources", [])
        context_text = result.get("context", {}).get("context", "")
        citations = _citation_payload(sources)

        yield _event(
            {
                "type": "citations",
                "citations": citations,
                "hit_count": len(citations),
            }
        )

        live_stream_enabled = os.environ.get("CHAT_STREAM_MODE", "auto").lower() != "simulated"
        can_stream_live = bool(os.environ.get("OPENAI_API_KEY")) and live_stream_enabled

        if can_stream_live:
            token_stream = _stream_live_generation(question, context_text)
        else:
            full_answer = generate_stage(question, context_text, live=False).get("answer", "")
            token_stream = _iter_text_chunks(full_answer)

        for chunk in token_stream:
            answer_buffer += chunk
            yield _event({"type": "delta", "content": chunk})
            if not can_stream_live:
                time.sleep(0.02)

        yield _event(
            {
                "type": "done",
                "answer": answer_buffer,
                "citations": citations,
            }
        )
    except Exception as exc:  # pragma: no cover - streaming failures are surfaced to UI
        yield _event(
            {
                "type": "error",
                "message": f"Streaming failed: {exc}",
                "partial_answer": answer_buffer,
                "citations": citations,
            }
        )


@app.post("/chat")
def chat_documents() -> Any:
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question") or payload.get("query") or "").strip()
    k = int(payload.get("k") or 5)

    if not question:
        return jsonify({"status": "error", "message": "A non-empty question is required."}), 400

    from pipeline.runner import run_pipeline

    try:
        result = run_pipeline(question, k=max(1, k), live=False)
        citations = _citation_payload(result.get("context", {}).get("sources", []))
        answer = result.get("answer", {}).get("answer", "")
        return jsonify(
            {
                "status": "ok",
                "question": question,
                "answer": answer,
                "citations": citations,
            }
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        return jsonify({"status": "error", "message": f"Chat failed: {exc}"}), 500


@app.post("/chat/stream")
def stream_chat_documents() -> Any:
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question") or payload.get("query") or "").strip()
    k = int(payload.get("k") or 5)

    if not question:
        return jsonify({"status": "error", "message": "A non-empty question is required."}), 400

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return Response(
        stream_with_context(_stream_chat_events(question, k=max(1, k))),
        mimetype="application/x-ndjson",
        headers=headers,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
