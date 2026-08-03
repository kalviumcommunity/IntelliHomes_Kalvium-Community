# Embeddings Report

```text
==============================================================
EMBEDDINGS FUNDAMENTALS & VECTOR REPRESENTATION
==============================================================

Mode     : live (default: Ollama + nomic-embed-text)
Endpoint : http://localhost:11434/v1
Model    : nomic-embed-text

--------------------------------------------------------------
TASK 1 — SAMPLE TEXTS -> EMBEDDINGS
--------------------------------------------------------------

'How do I reset my account password?'
  first 8 values : [+0.0125, +0.0157, -0.1330, +0.0345, +0.0651, +0.0023, +0.0084, +0.0291, …]

'Steps to recover access to my login'
  first 8 values : [+0.0284, -0.0197, -0.1482, +0.0030, +0.0404, -0.0435, +0.0310, +0.0590, …]

'The cafeteria menu has pasta today'
  first 8 values : [-0.0142, +0.0673, -0.1530, +0.0264, +0.0677, +0.0185, -0.0026, -0.0178, …]

'What documents do I need to transfer property ownership?'
  first 8 values : [+0.0261, -0.0129, -0.1259, -0.0091, +0.0145, -0.0009, -0.0328, +0.0681, …]

'Steps to transfer the title of a property to a new owner'
  first 8 values : [+0.0430, -0.0004, -0.1542, -0.0149, +0.0120, +0.0093, -0.0168, +0.0223, …]

--------------------------------------------------------------
TASK 2 — VECTOR DIMENSION
--------------------------------------------------------------

Dimension of embeddings[0] : 768
All texts same length       : True

--------------------------------------------------------------
TASK 3 — COSINE SIMILARITY (higher = closer meaning)
--------------------------------------------------------------

[0] 'How do I reset my account password?'
[1] 'Steps to recover access to my login'
[2] 'The cafeteria menu has pasta today'
[3] 'What documents do I need to transfer property ownership?'
[4] 'Steps to transfer the title of a property to a new owner'

similar    [0] vs [1]  : +0.6746
similar    [3] vs [4]  : +0.7734
dissimilar [0] vs [2]  : +0.3058
dissimilar [1] vs [2]  : +0.2996

Similar pair scores higher : True

==============================================================
TASK 4 — WHAT DO THESE VECTORS REPRESENT?
==============================================================

Each vector is a numeric representation of *meaning*. The model
was trained so that texts about the same topic land close together
in vector space and unrelated texts land far apart.

The numbers are NOT:
- random IDs — the same text always produces the same vector;
- keyword counts — "login" and "password" share no words, yet
  their vectors point in nearly the same direction.

They ARE a dense coordinate system for semantics: every dimension
responds to a pattern the model learned across billions of texts.
This is why RAG retrieval can match meaning, not just keywords.

In RAG each chunk is embedded and stored in a vector database.
A user question is embedded too, and retrieval becomes a
nearest-neighbor search over chunk vectors.


==============================================================
```
