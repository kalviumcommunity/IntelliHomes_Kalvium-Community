from prompts.renderer import (
    assemble_retrieved_context,
    build_augmented_prompt,
    build_citation_map,
    build_cited_answer,
)


def test_assemble_retrieved_context_adds_source_markers() -> None:
    chunks = [
        {
            "text": "A Title Deed confirms legal ownership.",
            "metadata": {"source": "property_guide.txt", "chunk_id": "property_guide-2", "section": "Section 2"},
        },
        {
            "text": "Property Tax Receipts confirm taxes are paid.",
            "metadata": {"source": "taxes.txt", "chunk_id": "taxes-1", "section": "Section 1"},
        },
    ]

    context = assemble_retrieved_context(chunks, token_budget=140, reserved_tokens=40)

    assert "[1] property_guide.txt" in context
    assert "chunk_id=property_guide-2" in context
    assert "[2] taxes.txt" in context
    assert "A Title Deed" in context


def test_build_augmented_prompt_uses_grounding_instructions() -> None:
    system_prompt, user_prompt = build_augmented_prompt(
        [{"text": "Ownership should be verified.", "metadata": {"source": "ownership.txt", "chunk_id": "ownership-1", "section": "Section 1"}}],
        "What should be verified?",
        token_budget=180,
        reserved_tokens=70,
    )

    assert "Use only the provided context" in system_prompt
    assert "[1] ownership.txt" in system_prompt
    assert "chunk_id=ownership-1" in system_prompt
    assert "What should be verified?" in user_prompt


def test_build_citation_map_uses_real_metadata() -> None:
    chunks = [{"text": "Legal ownership requires a title deed.", "metadata": {"source": "property_guide.txt", "chunk_id": "property_guide-2", "section": "Section 2", "page": 14, "position": 1}}]

    citation_map = build_citation_map(chunks)

    assert citation_map[1]["source"] == "property_guide.txt"
    assert citation_map[1]["chunk_id"] == "property_guide-2"
    assert citation_map[1]["section"] == "Section 2"
    assert citation_map[1]["page"] == 14


def test_build_cited_answer_rejects_fabricated_sources() -> None:
    chunks = [{"text": "A Title Deed confirms legal ownership.", "metadata": {"source": "property_guide.txt", "chunk_id": "property_guide-2", "section": "Section 2"}}]

    answer = build_cited_answer("The title deed proves ownership.", chunks, [1])
    assert "[1]" in answer
    assert "property_guide.txt" in answer
    assert "property_guide-2" in answer

    fallback = build_cited_answer("The title deed proves ownership.", [], [7])
    assert fallback == "I don't have enough information to answer confidently."
