def attach_metadata(chunks, source):
    metadata_chunks = []

    for index, chunk in enumerate(chunks):
        metadata_chunks.append({
            "text": chunk,
            "metadata": {
                "source": source,
                "section": f"Section {index + 1}",
                "position": index
            }
        })

    return metadata_chunks