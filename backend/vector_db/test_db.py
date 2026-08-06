import os
import random

import chromadb
from dotenv import load_dotenv

load_dotenv()

db_path = os.getenv("CHROMA_PATH", "chroma_db")
collection_name = os.getenv("COLLECTION_NAME", "property_chunks")

client = chromadb.PersistentClient(path=db_path)

# Drop any existing collection first so this demo never mixes its random
# vector with the real indexed corpus (dimension/space would mismatch).
client.delete_collection(collection_name)

collection = client.get_or_create_collection(
    collection_name,
    metadata={"hnsw:space": "cosine"},
    embedding_function=None,
)

# 768-dim to match the corpus embeddings (nomic-embed-text / simulated).
embedding = [random.random() for _ in range(768)]

collection.add(
    ids=["chunk_1"],
    embeddings=[embedding],
    documents=["A Title Deed confirms legal ownership."],
    metadatas=[
        {
            "source": "property_guide.txt",
            "section": "Section 1",
            "position": 0,
        }
    ],
)

result = collection.get(ids=["chunk_1"])

print(result)

print()

print("ID:", result["ids"][0])

print("Vector length:", len(embedding))

print("Text:", result["documents"][0])

print("Metadata:", result["metadatas"][0])
