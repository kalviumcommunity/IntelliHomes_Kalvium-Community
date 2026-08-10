import os

import chromadb
from dotenv import load_dotenv

from reranking.reranker import rerank

load_dotenv()

client = chromadb.PersistentClient(
    path=os.getenv("CHROMA_PATH", "chroma_db")
)

collection = client.get_collection(
    name="property_chunks"
)

query = "What documents should I verify before buying a property?"

CANDIDATE_COUNT = 10
FINAL_K = 3

print("===== RETRIEVAL + RE-RANKING =====")
print()
print("Query:")
print(query)
print()

# Initial retrieval
results = collection.query(
    query_texts=[query],
    n_results=CANDIDATE_COUNT
)

candidates = []

for i in range(len(results["ids"][0])):
    candidates.append({
        "id": results["ids"][0][i],
        "document": results["documents"][0][i],
        "metadata": results["metadatas"][0][i],
        "distance": results["distances"][0][i],
    })

print("===== INITIAL CANDIDATES =====")
print()

for i, candidate in enumerate(candidates, 1):
    print(f"Rank {i}")
    print(f"ID: {candidate['id']}")
    print(f"Vector score: {candidate['distance']:.4f}")
    print(f"Source: {candidate['metadata']['source']}")
    print(f"Metadata: {candidate['metadata']}")
    print(f"Text: {candidate['document']}")
    print()

# Re-ranking
reranked = rerank(query, candidates)

print("===== AFTER RE-RANKING =====")
print()

for i, candidate in enumerate(reranked, 1):
    print(f"Rank {i}")
    print(f"ID: {candidate['id']}")
    print(f"Original vector score: {candidate['distance']:.4f}")
    print(f"Re-rank score: {candidate['rerank_score']}")
    print(f"Matched words: {candidate['matched_words']}")
    print(f"Source: {candidate['metadata']['source']}")
    print(f"Metadata: {candidate['metadata']}")
    print(f"Text: {candidate['document']}")
    print()

# Final top 3
final_results = reranked[:FINAL_K]

print("===== FINAL TOP 3 =====")
print()

for i, candidate in enumerate(final_results, 1):
    print(f"Final Rank {i}")
    print(f"ID: {candidate['id']}")
    print(f"Source: {candidate['metadata']['source']}")
    print(f"Text: {candidate['document']}")
    print(f"Metadata: {candidate['metadata']}")
    print(f"Vector score: {candidate['distance']:.4f}")
    print(f"Re-rank score: {candidate['rerank_score']}")
    print()
