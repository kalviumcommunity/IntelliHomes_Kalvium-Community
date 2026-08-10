import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.vector_search import (  # noqa: E402
    hybrid_search,
    keyword_score,
    metadata_value_counts,
    open_collection,
    search,
)
from scripts.embed_corpus import embed_offline  # noqa: E402

# A small multi-category corpus, mirroring the shape of the real store but
# with categories so the metadata-filter and hybrid demos can be tested.
# Each document is one chunk; ``insp.txt`` carries an exact ID token.
DOCS = [
    ("legal.txt", "legal", "A Title Deed confirms legal ownership of the property."),
    ("tax.txt", "tax", "Property tax receipts confirm that all taxes have been paid."),
    (
        "tax2.txt",
        "tax",
        "Annual property tax payment records are kept with the deed for proof.",
    ),
    (
        "community.txt",
        "community",
        "The clubhouse is available for booking 48 hours in advance.",
    ),
    (
        "insp.txt",
        "inspection",
        "Inspection report ID-777 for 42 Maple Street found no issues.",
    ),
]


@pytest.fixture
def collection(tmp_path):
    """A ChromaDB collection indexed with the simulated 768-dim embedder."""
    client = __import__("chromadb").PersistentClient(path=str(tmp_path / "db"))
    _, col = open_collection(client=client)
    texts = [text for _, _, text in DOCS]
    col.add(
        ids=[f"{source}#0" for source, _, _ in DOCS],
        documents=texts,
        metadatas=[
            {"source": source, "section": "Section 1", "position": 0, "category": cat}
            for source, cat, _ in DOCS
        ],
        embeddings=embed_offline(texts),
    )
    return client, col


def _search(collection, query, k=10, **kwargs):
    client, col = collection
    return search(query, k, client=client, name=col.name, **kwargs)


def _hybrid(collection, query, k=10, **kwargs):
    client, col = collection
    return hybrid_search(query, k, client=client, name=col.name, **kwargs)


# ── Task 1: metadata filter ───────────────────────────────────────────────


def test_search_with_category_filter_restricts_results(collection):
    result = _search(collection, "clubhouse booking", metadata_filter={"category": "community"})
    assert result["total_chunks"] == 5
    assert result["total_matching"] == 1
    assert result["metadata_filter"] == {"category": "community"}
    assert result["k"] == 1  # clamped to the matching count
    assert {hit["id"] for hit in result["results"]} == {"community.txt#0"}
    assert all(hit["metadata"]["category"] == "community" for hit in result["results"])


def test_search_with_source_filter_restricts_results(collection):
    result = _search(collection, "property", metadata_filter={"source": "tax.txt"})
    assert {hit["id"] for hit in result["results"]} == {"tax.txt#0"}
    assert all(hit["metadata"]["source"] == "tax.txt" for hit in result["results"])


def test_unfiltered_search_has_no_filter_and_counts_everything(collection):
    result = _search(collection, "clubhouse", k=2)
    assert result["metadata_filter"] is None
    assert result["total_matching"] == result["total_chunks"] == 5


# ── Task 2: filtered vs unfiltered comparison ─────────────────────────────


def test_filtered_search_scopes_out_other_categories(collection):
    query = "how do I book the clubhouse for an event"
    plain = _search(collection, query, k=4)
    filtered = _search(collection, query, k=4, metadata_filter={"category": "community"})
    # Unfiltered returns chunks from every category…
    assert len(plain["results"]) == 4
    assert {hit["metadata"]["category"] for hit in plain["results"]} == {
        "legal",
        "tax",
        "community",
        "inspection",
    }
    # …while the filtered search only returns the matching category.
    assert {hit["id"] for hit in filtered["results"]} == {"community.txt#0"}
    assert all(hit["metadata"]["category"] == "community" for hit in filtered["results"])


def test_filter_improves_precision_over_plain_top_k(collection):
    query = "how do I prove property taxes were paid"
    relevant = {"tax.txt#0", "tax2.txt#0"}
    plain = _search(collection, query, k=2)
    filtered = _search(collection, query, k=2, metadata_filter={"category": "tax"})
    plain_hits = sum(1 for hit in plain["results"] if hit["id"] in relevant)
    filtered_hits = sum(1 for hit in filtered["results"] if hit["id"] in relevant)
    # Every filtered result is relevant; plain top-k can include legal noise.
    assert filtered_hits == 2
    assert filtered_hits >= plain_hits


# ── Task 3: keyword + hybrid matching ─────────────────────────────────────


def test_keyword_score_exact_terms():
    text = "Inspection report ID-777 for 42 Maple Street found no issues."
    assert keyword_score("ID-777", text) == 1.0
    assert keyword_score("clubhouse", "The clubhouse is available for booking.") == 1.0
    assert keyword_score("ID-777", "The clubhouse is available for booking.") == 0.0
    assert keyword_score("", text) == 0.0


def test_keyword_score_exact_phrase_bonus():
    text = "Inspection report ID-777 for 42 Maple Street found no issues."
    # Term overlap only: "structural" is missing -> 5 of 6 terms match.
    base = keyword_score(
        "ID-777 found no structural issues", text, exact_phrase_bonus=0.0
    )
    assert base == pytest.approx(5 / 6)
    # With a phrase bonus but no verbatim phrase, the score is unchanged.
    boosted = keyword_score(
        "ID-777 found no structural issues", text, exact_phrase_bonus=0.25
    )
    assert boosted == base
    # A verbatim phrase scores at least as high as term overlap and caps at 1.
    full = keyword_score("inspection report ID-777", text, exact_phrase_bonus=0.25)
    assert full == 1.0


def test_hybrid_search_ranks_exact_id_match_first(collection):
    hybrid = _hybrid(collection, "ID-777")
    assert hybrid["results"][0]["id"] == "insp.txt#0"
    top = hybrid["results"][0]
    assert top["keyword_score"] == 1.0  # exact term present
    assert top["vector_score"] >= 0.0
    # The keyword boost raises the combined score above the vector score.
    assert top["score"] > top["vector_score"]
    assert hybrid["hybrid"]["candidates_scored"] >= hybrid["k"]


def test_hybrid_search_reports_component_scores(collection):
    hybrid = _hybrid(collection, "clubhouse booking hours", k=3)
    assert len(hybrid["results"]) == 3
    for hit in hybrid["results"]:
        assert "vector_score" in hit and "keyword_score" in hit and "score" in hit
        assert hit["score"] == pytest.approx(
            hit["vector_score"] + 0.3 * hit["keyword_score"], abs=1e-6
        )


def test_hybrid_search_respects_metadata_filter(collection):
    hybrid = _hybrid(
        collection, "inspection", metadata_filter={"category": "inspection"}
    )
    assert {hit["id"] for hit in hybrid["results"]} == {"insp.txt#0"}
    assert hybrid["total_matching"] == 1


# ── Task 1: metadata introspection ────────────────────────────────────────


def test_metadata_value_counts_per_key(collection):
    client, col = collection
    counts = metadata_value_counts(client=client, name=col.name, key="category")
    assert counts == {"legal": 1, "tax": 2, "community": 1, "inspection": 1}


def test_metadata_value_counts_nested(collection):
    client, col = collection
    nested = metadata_value_counts(client=client, name=col.name)
    assert nested["category"]["tax"] == 2
    assert nested["source"]["legal.txt"] == 1
    assert nested["section"] == {"Section 1": 5}
