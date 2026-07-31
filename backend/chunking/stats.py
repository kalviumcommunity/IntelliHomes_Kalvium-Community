def chunk_stats(chunks):
    count = len(chunks)
    avg = sum(len(chunk) for chunk in chunks) / count if count else 0
    return count, avg