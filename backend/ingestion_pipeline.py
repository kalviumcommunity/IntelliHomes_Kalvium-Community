import os

from chunking.strategies import paragraph_chunk
from chunking.metadata import attach_metadata

DOCUMENT_FOLDER = "documents"

successful = 0
failed = 0
total_chunks = 0

sample_chunks = []

files = os.listdir(DOCUMENT_FOLDER)

print("===== INGESTION STARTED =====\n")

for filename in files:

    path = os.path.join(DOCUMENT_FOLDER, filename)

    try:

        with open(path, encoding="utf8") as f:
            text = f.read()

        # Cleaning
        cleaned = text.strip()

        # Chunking
        chunks = paragraph_chunk(cleaned)

        # Metadata
        tagged = attach_metadata(chunks, filename)

        successful += 1
        total_chunks += len(tagged)

        sample_chunks.extend(tagged[:2])

        print(f"✓ {filename}")

    except Exception:

        failed += 1

        print(f"✗ {filename}")

print("\n===== SUMMARY =====")

print(f"Source documents : {len(files)}")
print(f"Ingested         : {successful}")
print(f"Failures         : {failed}")
print(f"Total chunks     : {total_chunks}")

print("\n===== SAMPLE CHUNKS =====")

for chunk in sample_chunks:
    print(chunk)
    print()