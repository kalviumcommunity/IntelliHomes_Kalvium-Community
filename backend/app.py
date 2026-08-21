from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from flask import (
    Flask,
    Response,
    jsonify,
    request,
    send_from_directory,
    stream_with_context,
)
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

BACKEND_ROOT = Path(__file__).resolve().parent
FRONTEND_DIST = BACKEND_ROOT.parent / "frontend" / "dist"
app = Flask(__name__, static_folder=str(FRONTEND_DIST), static_url_path="/assets")

PROPERTIES = [
    {
        "id": "north-harbor",
        "name": "North Harbor Residence",
        "detail": "4 bed • 3 bath • 3,250 sq ft",
        "price": "$1.48M",
        "color": "blue",
        "tag": "Shortlisted",
        "bedrooms": 4,
        "mode": "Buy",
        "score": 92,
    },
    {
        "id": "willow-terrace",
        "name": "Willow Terrace",
        "detail": "3 bed • 2 bath • 2,180 sq ft",
        "price": "$1.02M",
        "color": "green",
        "tag": "Low risk",
        "bedrooms": 3,
        "mode": "Buy",
        "score": 88,
    },
    {
        "id": "modern-loft",
        "name": "Modern Loft",
        "detail": "2 bed • 2 bath • 1,420 sq ft",
        "price": "$1.24M",
        "color": "blue",
        "tag": "AI match",
        "bedrooms": 2,
        "mode": "Invest",
        "score": 84,
    },
    {
        "id": "garden-villa",
        "name": "Garden Villa",
        "detail": "4 bed • 4 bath • 3,900 sq ft",
        "price": "$1.71M",
        "color": "green",
        "tag": "Top pick",
        "bedrooms": 4,
        "mode": "Buy",
        "score": 95,
    },
]

DATABASE_PATH = Path(
    os.environ.get("INTELLIHOMES_DB_PATH", BACKEND_ROOT / "intellihomes.sqlite3")
)


def _database() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _init_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _database() as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS properties (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                detail TEXT NOT NULL,
                price TEXT NOT NULL,
                color TEXT NOT NULL,
                tag TEXT NOT NULL,
                bedrooms INTEGER NOT NULL,
                mode TEXT NOT NULL,
                score INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                source TEXT NOT NULL,
                chunks_indexed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """)
        existing = connection.execute(
            "SELECT COUNT(*) AS count FROM properties"
        ).fetchone()["count"]
        if not existing:
            connection.executemany(
                "INSERT INTO properties (id, name, detail, price, color, tag, bedrooms, mode, score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        item["id"],
                        item["name"],
                        item["detail"],
                        item["price"],
                        item["color"],
                        item["tag"],
                        item["bedrooms"],
                        item["mode"],
                        item["score"],
                    )
                    for item in PROPERTIES
                ],
            )


_init_database()


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


RUNTIME_DIR = Path(os.environ.get("INTELLIHOMES_RUNTIME_DIR", BACKEND_ROOT / "runtime"))
UPLOAD_DIR = Path(os.environ.get("INTELLIHOMES_UPLOAD_DIR", RUNTIME_DIR / "uploads"))
EMBEDDING_DIR = Path(
    os.environ.get("INTELLIHOMES_EMBEDDING_DIR", RUNTIME_DIR / "embeddings")
)
MAX_UPLOAD_BYTES = int(
    os.environ.get("INTELLIHOMES_MAX_UPLOAD_BYTES", str(25 * 1_048_576))
)
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


@app.get("/properties")
def properties() -> Any:
    search_mode = request.args.get("mode", "Buy").strip().lower()
    minimum_bedrooms = request.args.get("bedrooms", "3+").replace("+", "")
    search_text = request.args.get("q", "").strip().lower()
    try:
        bedroom_floor = int(minimum_bedrooms)
    except ValueError:
        bedroom_floor = 3
    with _database() as connection:
        results = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM properties WHERE bedrooms >= ?", (bedroom_floor,)
            ).fetchall()
        ]
    if search_mode != "all":
        results = [
            property_item
            for property_item in results
            if property_item["mode"].lower() == search_mode or search_mode == "rent"
        ]
    if search_text:
        results = [
            property_item
            for property_item in results
            if search_text in property_item["name"].lower()
            or search_text in property_item["detail"].lower()
        ]
    return jsonify({"status": "ok", "count": len(results), "properties": results})


@app.post("/compare")
def compare_properties() -> Any:
    payload = request.get_json(silent=True) or {}
    property_ids = payload.get("property_ids") or []
    if not isinstance(property_ids, list) or not property_ids:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "property_ids must contain at least one property.",
                }
            ),
            400,
        )
    placeholders = ",".join("?" for _ in property_ids)
    with _database() as connection:
        rows = connection.execute(
            f"SELECT * FROM properties WHERE id IN ({placeholders})", property_ids
        ).fetchall()
    comparisons = []
    for row in rows:
        item = dict(row)
        item["legal_score"] = min(99, item["score"] + 3)
        item["location_score"] = min(99, item["score"] - 1)
        item["recommendation"] = "Strong" if item["score"] >= 88 else "Review documents"
        comparisons.append(item)
    return jsonify({"status": "ok", "properties": comparisons})


@app.get("/market-stats")
def market_stats() -> Any:
    return jsonify(
        {
            "status": "ok",
            "change": "+12.8% Q3",
            "values": [30, 45, 59, 73, 53, 68],
            "labels": ["Jan", "Mar", "May", "Jul", "Sep"],
        }
    )


@app.get("/documents")
def documents() -> Any:
    with _database() as connection:
        rows = connection.execute(
            "SELECT filename, source, chunks_indexed, created_at FROM documents ORDER BY created_at DESC LIMIT 20",
        ).fetchall()
    return jsonify({"status": "ok", "documents": [dict(row) for row in rows]})


@app.get("/")
def frontend_index() -> Any:
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return send_from_directory(FRONTEND_DIST, "index.html")
    return jsonify(
        {
            "message": "IntelliHomes backend is running",
            "frontend": "Run npm run dev in frontend/",
        }
    )


def _safe_upload_path(filename: str) -> Path:
    safe_name = secure_filename(filename)
    if not safe_name:
        raise ValueError("file name is required")

    suffix = Path(safe_name).suffix.lower()
    stem = Path(safe_name).stem or "upload"
    unique = uuid.uuid4().hex[:8]
    target = UPLOAD_DIR / f"{stem}-{unique}{suffix}"
    return target


def _build_store_for_document(
    doc_path: Path, doc_source: str, *records: dict[str, Any]
) -> dict[str, Any]:
    text_records = [r for r in records if r.get("text")]
    vectors = [r.get("vector") for r in text_records]
    valid_vectors = [v for v in vectors if v is not None]
    if not text_records or not valid_vectors:
        raise ValueError("uploaded document did not produce any valid chunk vectors")

    dim = len(valid_vectors[0])
    if any(len(v) != dim for v in valid_vectors):
        raise ValueError("embedding dimension mismatch across uploaded chunks")

    tokens = estimate_tokens([r["text"] for r in text_records])
    _, cost = estimate_cost(
        tokens, os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
    )
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
        "generated_at": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(timespec="seconds"),
        "output_file": str(
            EMBEDDING_DIR
            / f"{Path(doc_source).stem}-{uuid.uuid4().hex[:8]}-embeddings.json"
        ),
        "records": text_records,
    }
    return store


def _process_upload(file_path: Path) -> dict[str, Any]:
    doc = load_file(file_path)
    records = chunk_corpus([doc])
    if not records:
        raise ValueError("uploaded document produced no searchable chunks")

    texts = [record["text"] for record in records]
    mode, vectors, stats = embed_corpus_chunks(
        texts, use_live=os.environ.get("EMBEDDING_MODE", "").lower() != "simulated"
    )
    for record, vector in zip(records, vectors):
        if vector is not None:
            record["vector"] = vector

    records = [record for record in records if record.get("vector") is not None]
    if not records:
        raise ValueError("uploaded document could not be embedded")

    store_path = (
        EMBEDDING_DIR
        / f"{Path(doc.source).stem}-{uuid.uuid4().hex[:8]}-embeddings.json"
    )
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
        "cost_usd": round(
            estimate_cost(
                estimate_tokens(texts),
                os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"),
            )[1],
            6,
        ),
        "generated_at": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(timespec="seconds"),
        "output_file": str(store_path),
        "records": records,
    }
    save_store(store, store_path)
    report = index_store(store_path=store_path, rebuild=False)
    with _database() as connection:
        connection.execute(
            "INSERT INTO documents (filename, source, chunks_indexed) VALUES (?, ?, ?)",
            (file_path.name, doc.source, report["indexed"]),
        )

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
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Unsupported file format: {suffix or 'unknown'}",
                }
            ),
            415,
        )

    bytes_data = file_obj.read()
    if not bytes_data or not bytes_data.strip():
        return jsonify({"status": "error", "message": "Uploaded file is empty."}), 400

    if len(bytes_data) > MAX_UPLOAD_BYTES:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"File exceeds the {MAX_UPLOAD_BYTES} byte upload limit.",
                }
            ),
            413,
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_path = _safe_upload_path(filename)
    safe_path.write_bytes(bytes_data)

    try:
        result = _process_upload(safe_path)
        return (
            jsonify(
                {
                    "status": "uploaded",
                    "message": "Document uploaded and indexed successfully.",
                    **result,
                }
            ),
            201,
        )
    except UnsupportedFormatError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 415
    except (DocumentLoadError, ValueError, OSError) as exc:
        return (
            jsonify(
                {"status": "error", "message": f"Document processing failed: {exc}"}
            ),
            400,
        )
    except (
        Exception
    ) as exc:  # pragma: no cover - defensive handler for runtime failures
        return (
            jsonify(
                {"status": "error", "message": f"Unexpected processing error: {exc}"}
            ),
            500,
        )


@app.post("/query")
def query_documents() -> Any:
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("query") or payload.get("question") or "").strip()
    if not question:
        return (
            jsonify({"status": "error", "message": "A non-empty query is required."}),
            400,
        )

    from retrieval.vector_search import search

    try:
        result = search(question, k=5)
    except Exception as exc:  # pragma: no cover - retrieval runtime guard
        return jsonify({"status": "error", "message": f"Query failed: {exc}"}), 500

    return jsonify(
        {
            "status": "ok",
            "query": question,
            "results": [
                {
                    "id": hit.get("id"),
                    "score": hit.get("score"),
                    "source": hit.get("metadata", {}).get("source"),
                    "section": hit.get("metadata", {}).get("section"),
                    "chunk_id": hit.get("metadata", {}).get("chunk_id")
                    or hit.get("id"),
                    "text": hit.get("text"),
                }
                for hit in result.get("results", [])
            ],
        }
    )


@app.post("/answer")
def answer_question() -> Any:
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("query") or payload.get("question") or "").strip()
    if not question:
        return (
            jsonify(
                {"status": "error", "message": "A non-empty question is required."}
            ),
            400,
        )

    from retrieval.vector_search import search

    try:
        result = search(question, k=5)
    except Exception as exc:  # pragma: no cover - retrieval runtime guard
        return (
            jsonify({"status": "error", "message": f"Answer generation failed: {exc}"}),
            500,
        )

    hits = result.get("results", [])
    if not hits:
        return jsonify(
            {
                "status": "ok",
                "query": question,
                "answer": "I don't have enough information to answer confidently.",
                "sources": [],
            }
        )

    sources = []
    for hit in hits:
        metadata = hit.get("metadata", {})
        source = metadata.get("source") or hit.get("id")
        sources.append(
            {
                "id": hit.get("id"),
                "source": source,
                "chunk_id": metadata.get("chunk_id") or hit.get("id"),
                "section": metadata.get("section"),
                "score": hit.get("score"),
                "text": hit.get("text"),
            }
        )

    top_text = hits[0].get("text", "").strip()
    if not top_text:
        answer = "I don't have enough information to answer confidently."
    else:
        source_names = ", ".join(s["source"] for s in sources[:3])
        answer = f"Based on the retrieved source(s) ({source_names}), the relevant context says: {top_text}"

    return jsonify(
        {
            "status": "ok",
            "query": question,
            "answer": answer,
            "sources": sources,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
