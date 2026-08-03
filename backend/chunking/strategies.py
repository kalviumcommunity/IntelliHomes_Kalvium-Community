from dataclasses import dataclass

try:
    import tiktoken
except ImportError:  # pragma: no cover - environment fallback
    tiktoken = None


@dataclass
class TokenChunk:
    text: str
    token_count: int
    overlap_tokens: int


def _get_encoder(model_name: str = "gpt-4o"):
    if tiktoken is None:
        return None

    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def fixed_chunk(text: str, size: int = 100, overlap: int = 20):
    return [
        text[start:end]
        for start, end in _window_ranges(len(text), size, overlap)
    ]


def token_chunk(text: str, chunk_size: int = 180, overlap: int = 30, model_name: str = "gpt-4o") -> list[TokenChunk]:
    encoder = _get_encoder(model_name)
    if encoder is not None:
        tokens = encoder.encode(text)
        if not tokens:
            return []

        chunks: list[TokenChunk] = []
        start = 0
        step = max(1, chunk_size - overlap)

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = encoder.decode(chunk_tokens)
            chunks.append(
                TokenChunk(
                    text=chunk_text.decode("utf-8"),
                    token_count=len(chunk_tokens),
                    overlap_tokens=overlap,
                )
            )
            start += step

        return chunks

    # Fallback: estimate tokens as roughly 4 characters per token for English text.
    approx_tokens = max(1, len(text) // 4)
    if approx_tokens <= 0:
        return []

    chunks: list[TokenChunk] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = min(start + chunk_size * 4, len(text))
        chunk_text = text[start:end]
        chunks.append(
            TokenChunk(
                text=chunk_text,
                token_count=max(1, len(chunk_text) // 4),
                overlap_tokens=overlap,
            )
        )
        start += step * 4

    return chunks


def _window_ranges(length: int, size: int, overlap: int) -> list[tuple[int, int]]:
    if size <= 0:
        return []

    step = max(1, size - overlap)
    windows: list[tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(start + size, length)
        windows.append((start, end))
        start += step
    return windows


def paragraph_chunk(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]