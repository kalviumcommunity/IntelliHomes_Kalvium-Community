from prompts.renderer import assemble_retrieved_context, build_augmented_prompt


def test_assemble_retrieved_context_adds_source_markers() -> None:
    chunks = [
        {
            "text": "A Title Deed confirms legal ownership.",
            "metadata": {"source": "property_guide.txt"},
        },
        {
            "text": "Property Tax Receipts confirm taxes are paid.",
            "metadata": {"source": "taxes.txt"},
        },
    ]

    context = assemble_retrieved_context(chunks, token_budget=140, reserved_tokens=40)

    assert "[1] property_guide.txt" in context
    assert "[2] taxes.txt" in context
    assert "A Title Deed" in context


def test_build_augmented_prompt_uses_grounding_instructions() -> None:
    system_prompt, user_prompt = build_augmented_prompt(
        [{"text": "Ownership should be verified.", "metadata": {"source": "ownership.txt"}}],
        "What should be verified?",
        token_budget=180,
        reserved_tokens=70,
    )

    assert "Use only the provided context" in system_prompt
    assert "[1] ownership.txt" in system_prompt
    assert "What should be verified?" in user_prompt
