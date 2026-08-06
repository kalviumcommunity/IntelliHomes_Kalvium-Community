"""
index_corpus.py — Index the embeddings vector store into ChromaDB.

IntelliHomes RAG pipeline — Vector-database indexing stage.

Reads the JSON vector store produced by embed_corpus.py (one record per
chunk: id / text / metadata / vector) and writes every record into the
ChromaDB collection used by the retrieval stage, so top-k semantic search
can run against the corpus.

The collection is rebuilt by default — deleted and recreated with the
cosine space and no default embedding function (this pipeline supplies its
own vectors) — so the index always matches the store's vector dimension
and metric. Set INDEX_REBUILD=0 to upsert into the existing collection.

Environment:
    CHROMA_PATH        ChromaDB persist directory (default: chroma_db).
    COLLECTION_NAME    Collection name (default: property_chunks).
    EMBEDDINGS_STORE   Path of the embeddings JSON store (default:
                       ../data/embeddings/cleaned_corpus-embeddings.json).
    INDEX_REBUILD      1 = delete + recreate the collection (default),
                       0 = upsert into the existing collection.
    INDEX_BATCH_SIZE   Records per add() call (default: 64).

Usage
-----
    python scripts/index_corpus.py            # from backend/
    python scripts/index_corpus.py --no-rebuild
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Make `backend/` importable no matter which directory the script is run from.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import chromadb  # noqa: E402

from retrieval.vector_search import (  # noqa: E402
    COLLECTION_NAME,
    default_store_path,
    open_collection,
)

# ── Configuration (everything from the environment) ───────────────────────

REBUILD = os.environ.get("INDEX_REBUILD", "1").lower() not in ("0", "false", "no")
BATCH_SIZE = int(os.environ.get("INDEX_BATCH_SIZE", "64"))


def load_records(store: dict) -> tuple[list[dict], list[str]]:
    """Validate store records; return ``(valid_records, skipped_ids)``.

    A record is indexable when it carries an id, source text, metadata and a
    non-empty vector — everything ChromaDB needs for retrieval.
    """
    valid: list[dict] = []
    skipped: list[str] = []
    for record in store.get("records", []):
        vector = record.get("vector") or []
        if (
            record.get("id")
            and record.get("text")
            and record.get("metadata") is not None
            and vector
        ):
            valid.append(record)
        else:
            skipped.append(str(record.get("id", "?")))
    return valid, skipped


def index_store(
    store_path: str | Path | None = None,
    *,
    rebuild: bool | None = None,
    client: chromadb.ClientAPI | None = None,
    path: str | None = None,
    name: str | None = None,
) -> dict:
    """Index the embeddings store at *store_path* into ChromaDB.

    Returns a report dict with the collection, space, dimension, counts and
    any records that could not be indexed.
    """
    if store_path is None:
        store_path = default_store_path()
    store_path = Path(store_path)

    from scripts.embed_corpus import load_store

    store = load_store(store_path)
    records, skipped = load_records(store)
    if not records:
        raise ValueError(f"embeddings store has no indexable records: {store_path}")

    client, collection = open_collection(client=client, path=path, name=name)
    collection_name = collection.name

    rebuild = REBUILD if rebuild is None else rebuild
    if rebuild:
        # Drop any previous collection: it may hold stale test vectors or a
        # different dimension/space, which would corrupt retrieval.
        client.delete_collection(collection_name)
        collection = client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        print(
            f"Rebuilding collection '{collection_name}' "
            f"(cosine space, no default embedding function)"
        )

    indexed = 0
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        collection.upsert(
            ids=[r["id"] for r in batch],
            documents=[r["text"] for r in batch],
            metadatas=[r["metadata"] for r in batch],
            embeddings=[r["vector"] for r in batch],
        )
        indexed += len(batch)

    return {
        "collection": collection_name,
        "space": "cosine",
        "dim": len(records[0]["vector"]),
        "rebuild": rebuild,
        "mode": store.get("mode"),
        "model": store.get("model"),
        "records_in_store": len(records),
        "indexed": indexed,
        "skipped": skipped,
        "count": collection.count(),
        "store": str(store_path),
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    rebuild = REBUILD
    if "--no-rebuild" in args:
        rebuild = False

    try:
        report = index_store(rebuild=rebuild)
    except FileNotFoundError as exc:
        print(f"ERROR: embeddings store not found — {exc}")
        return 1
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("\nINDEX REPORT")
    print("============")
    print(f"Collection        : {report['collection']}")
    print(f"Space             : {report['space']} (distance = 1 - cosine similarity)")
    print(f"Dimension         : {report['dim']}")
    print(f"Embedding mode    : {report['mode']} ({report['model']})")
    print(f"Rebuilt           : {report['rebuild']}")
    print(f"Records in store  : {report['records_in_store']}")
    print(f"Indexed           : {report['indexed']}")
    print(f"Skipped (invalid) : {len(report['skipped'])}")
    if report["skipped"]:
        print(f"  skipped ids: {report['skipped']}")
    print(f"Chunks in DB      : {report['count']}")
    print(f"Store             : {report['store']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
