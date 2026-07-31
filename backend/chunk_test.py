from chunking.strategies import fixed_chunk, paragraph_chunk
from chunking.stats import chunk_stats

with open("sample_document.txt", encoding="utf8") as f:
    document = f.read()

fixed = fixed_chunk(document)
paragraphs = paragraph_chunk(document)

print("===== FIXED SIZE CHUNKS =====")
for i, chunk in enumerate(fixed, 1):
    print(f"\nChunk {i}")
    print(chunk)

print("\n===== PARAGRAPH CHUNKS =====")
for i, chunk in enumerate(paragraphs, 1):
    print(f"\nChunk {i}")
    print(chunk)

fixed_count, fixed_avg = chunk_stats(fixed)
para_count, para_avg = chunk_stats(paragraphs)

print("\n===== STATISTICS =====")
print(f"Fixed-size: {fixed_count} chunks | Average size: {fixed_avg:.2f} characters")
print(f"Paragraph: {para_count} chunks | Average size: {para_avg:.2f} characters")