def attach_metadata(chunks, source, category=None):
    """Attach retrieval metadata (source/section/position) to each chunk.

    *category* is optional and only added when truthy, so existing callers
    (and their tests) keep the exact ``{source, section, position}`` shape.
    A category lets retrieval scope searches to a document type or topic
    (e.g. a ``where={"category": "legal"}`` ChromaDB filter).
    """
    metadata_chunks = []

    for index, chunk in enumerate(chunks):
        metadata = {
            "source": source,
            "section": f"Section {index + 1}",
            "position": index
        }
        if category:
            metadata["category"] = category
        metadata_chunks.append({
            "text": chunk,
            "metadata": metadata
        })

    return metadata_chunks