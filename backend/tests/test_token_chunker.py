import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chunking.strategies import token_chunk, paragraph_chunk


def test_token_chunk_uses_overlap_and_reports_tokens():
    text = "A title deed confirms ownership. A survey plan confirms boundaries. A tax receipt confirms payment history."

    chunks = token_chunk(text, chunk_size=12, overlap=4, model_name="gpt-4o")

    assert len(chunks) >= 2
    assert chunks[0].text != chunks[1].text
    assert chunks[0].overlap_tokens == 4
    assert chunks[0].token_count > 0


def test_paragraph_chunk_still_returns_paragraphs():
    text = "First paragraph.\n\nSecond paragraph."

    chunks = paragraph_chunk(text)

    assert len(chunks) == 2
