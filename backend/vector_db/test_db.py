import os
import random

import chromadb
from dotenv import load_dotenv

load_dotenv()

client = chromadb.PersistentClient(
    path=os.getenv("CHROMA_PATH")
)

collection = client.get_or_create_collection(
    "property_chunks"
)

embedding = [
    random.random()
    for _ in range(384)
]

collection.add(

    ids=["chunk_1"],

    embeddings=[embedding],

    documents=[
        "A Title Deed confirms legal ownership."
    ],

    metadatas=[

        {
            "source": "property_guide.txt",
            "section": "Section 2",
            "chunk_index": 1
        }

    ]
)

result = collection.get(ids=["chunk_1"])

print(result)

print()

print("ID:", result["ids"][0])

print("Vector length:", len(embedding))

print("Text:", result["documents"][0])

print("Metadata:", result["metadatas"][0])