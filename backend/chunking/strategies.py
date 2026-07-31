from textwrap import wrap


def fixed_chunk(text, size=100, overlap=20):
    chunks = []

    start = 0

    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap

    return chunks


def paragraph_chunk(text):
    return [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]