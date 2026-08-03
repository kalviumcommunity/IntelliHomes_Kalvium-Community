# Embedding Vectors — What They Represent

An **embedding vector** is a list of numbers that represents the _meaning_ of a
piece of text. The embedding model was trained on billions of sentences so that
the position of each vector in vector space encodes semantics:

- Texts about the **same topic** produce vectors that point in nearly the
  **same direction** (high cosine similarity).
- Texts about **unrelated topics** land **far apart** (low similarity).

## What the numbers are NOT

- **Not random IDs.** The same text always maps to the same vector — the
  mapping is deterministic for a given model.
- **Not keyword counts.** In the demo, "password" and "login" share no words,
  yet their vectors score `+0.67` similarity. A bag-of-words counter would
  score them near zero.
- **Not human-interpretable coordinates.** You cannot read one dimension in
  isolation (e.g. "dimension 42 = price"). Each dimension responds to a subtle
  pattern the model learned; the _full pattern_ carries the meaning.

## What they ARE

A **dense coordinate system for semantics**. Think of each vector as a point on
a high-dimensional sphere (768 dimensions for `nomic-embed-text`, 1536 for
`text-embedding-3-small`). Meaning is the _position_, and similarity is the
_direction_ between positions.

## Why this powers semantic search in RAG

In the IntelliHomes RAG pipeline:

1. Every cleaned chunk is embedded and stored in a vector database.
2. A user question ("How do I reset my password?") is embedded with the same
   model.
3. Retrieval becomes a **nearest-neighbor search**: find the chunk vectors
   closest to the question vector — even when the chunk says "account recovery
   steps" and never mentions "password".

Keyword matching fails on paraphrase; embedding similarity does not.
