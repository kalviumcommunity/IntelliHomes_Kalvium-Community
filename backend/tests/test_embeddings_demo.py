import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.embeddings_demo import (
    TEXTS,
    build_report,
    cosine,
    simulated_embed,
)


def test_cosine_similarity_extremes():
    # identical vectors -> 1.0, orthogonal -> 0.0, opposite -> -1.0
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert abs(cosine([1.0, 0.0], [0.0, 1.0])) < 1e-12
    assert abs(cosine([1.0, 0.0], [-1.0, 0.0]) + 1.0) < 1e-12


def test_cosine_zero_vector_does_not_crash():
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_simulated_embeddings_have_uniform_dimension():
    embeddings = simulated_embed(TEXTS)
    dims = {len(e) for e in embeddings}
    assert len(dims) == 1  # every text produces a vector of the same length
    assert dims.pop() == 768


def test_simulated_embeddings_are_deterministic():
    first = simulated_embed(TEXTS)
    second = simulated_embed(TEXTS)
    assert first == second


def test_similar_pair_scores_higher_than_dissimilar():
    embeddings = simulated_embed(TEXTS)
    similar = cosine(embeddings[0], embeddings[1])
    dissimilar = cosine(embeddings[0], embeddings[2])
    assert similar > dissimilar


def test_property_similar_pair_scores_higher():
    embeddings = simulated_embed(TEXTS)
    similar = cosine(embeddings[3], embeddings[4])
    dissimilar = cosine(embeddings[0], embeddings[2])
    assert similar > dissimilar


def test_report_reports_dimension_ranking_and_explainer():
    embeddings = simulated_embed(TEXTS)
    report = build_report(embeddings, live=False)
    assert "Dimension of embeddings[0] : 768" in report
    assert "All texts same length       : True" in report
    assert "Similar pair scores higher : True" in report
    assert "numeric representation of *meaning*" in report
