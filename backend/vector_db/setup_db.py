import os

from dotenv import load_dotenv
import chromadb

load_dotenv()

db_path = os.getenv("CHROMA_PATH")

client = chromadb.PersistentClient(path=db_path)

collection = client.get_or_create_collection(
    name="property_chunks"
)

print("Connected to ChromaDB")
print("Collection:", collection.name)