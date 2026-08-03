from chunking.strategies import paragraph_chunk
from chunking.metadata import attach_metadata

with open("sample_document.txt", encoding="utf8") as f:
    document = f.read()

chunks = paragraph_chunk(document)

metadata_chunks = attach_metadata(
    chunks,
    "sample_document.txt"
)

print("===== CHUNKS WITH METADATA =====\n")

for chunk in metadata_chunks:
    print(chunk)
    print()