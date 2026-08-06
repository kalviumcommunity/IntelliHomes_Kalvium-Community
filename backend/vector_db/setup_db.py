import os

from dotenv import load_dotenv
import chromadb

load_dotenv()

db_path = os.getenv("CHROMA_PATH", "chroma_db")
collection_name = os.getenv("COLLECTION_NAME", "property_chunks")

client = chromadb.PersistentClient(path=db_path)

# The retrieval pipeline supplies its own vectors (produced by
# embed_corpus.py), so no default embedding function is configured — ChromaDB
# must never embed text itself. The cosine space matches the similarity
# metric used for ranking (score = 1 - cosine distance).
collection = client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"},
    embedding_function=None,
)

print("Connected to ChromaDB")
print("Collection:", collection.name)
print("Space: cosine (distance = 1 - cosine similarity)")
print("Chunks indexed:", collection.count())
