# Batch Embedding Pipeline — Sample Run Summary

`backend/scripts/embed_corpus.py` embeds a prepared corpus in **batches**, retries
**transient API failures with exponential backoff**, reports **totals and an
approximate cost**, and **skips chunks that already have embeddings** on re-runs so
you never pay twice for the same text.

This report documents two real runs against the prepared corpus
(`backend/cleaned_corpus/`, embedded live with `nomic-embed-text` on local Ollama,
768-dim).

---

## Run 1 — full embed

Fresh run, no previous store: every chunk is new, so all 7 chunks are embedded in
a single batch (batch size 64).

```
Chunks: 7 total, 7 to embed, 0 skipped (already cached)
Embedded 7 chunks (live: nomic-embed-text)
```

| Metric                     | Value                                        |
| -------------------------- | -------------------------------------------- |
| Total chunks               | 7                                            |
| Embedded                   | 7                                            |
| Skipped (cached)           | 0                                            |
| Failed                     | 0                                            |
| Retries (live)             | 0                                            |
| Batches                    | 1 (batch size 64)                            |
| Mode / model               | live (`nomic-embed-text`)                    |
| Approx. tokens             | 60                                           |
| Price / 1M tokens          | $0.0000 (local model — assumed free)         |
| Approx. cost               | **$0.00**                                    |
| Store                      | `data/embeddings/cleaned_corpus-embeddings.json` |

---

## Run 2 — re-run, everything skipped

Same corpus, same store, run again. Every chunk already has an embedding whose
`id` **and** `text` match, so **no API call is made at all** and the run costs
nothing.

```
Chunks: 7 total, 0 to embed, 7 skipped (already cached)
```

| Metric                     | Value                                        |
| -------------------------- | -------------------------------------------- |
| Total chunks               | 7                                            |
| Embedded                   | 0                                            |
| Skipped (cached)           | 7                                            |
| Failed                     | 0                                            |
| Retries (live)             | 0                                            |
| Batches                    | 0 (batch size 64)                            |
| Mode / model               | cached (`nomic-embed-text`)                  |
| Approx. tokens             | 0                                            |
| Approx. cost               | **$0.00**                                    |

If run 2 had naively re-embedded everything it would have re-sent all 7 chunks to
the API for the same vectors — the skip avoids that duplicate work.

---

## How batching works (Task 1)

Chunks are grouped into requests of `EMBEDDING_BATCH_SIZE` (default **64**) instead
of one API call per chunk:

```
for start in range(0, len(texts), batch_size):
    batch = texts[start:start + batch_size]
    response = client.embeddings.create(model=model, input=batch)
```

7 chunks → **1 request**. A 10 000-chunk corpus → ~157 requests instead of 10 000.
This cuts round-trip latency and rate-limit pressure substantially.

## How retries and failures are handled (Task 2)

Each batch is sent through a retry loop with **exponential backoff + jitter**:

* Rate limits (`429`) and server errors (`5xx`) are retried up to
  `EMBEDDING_MAX_RETRIES` (default **5**) times.
* The delay doubles per attempt — `base_delay * 2 ** n` plus a small random jitter —
  starting at `EMBEDDING_RETRY_BASE_DELAY` (default **1.0 s**).
* Transport errors (connection refused, timeouts) are also retried.
* **Permanent errors** (`400/401/403`) are *not* retried — they raise immediately.
* A batch that exhausts its retries is **marked as failed**: its positions become
  `None`, the chunks are excluded from the store, and the failure appears in the
  run summary (`Failed`, and `Failed batches at: [start, …]`).

Example log line from a retry:

```
batch retry 2/5 after 4.0s — RateLimitError('HTTP 429')
```

## How cost is estimated (Task 3)

Tokens are counted with `tiktoken` (`cl100k_base`, the same BPE family as OpenAI's
embedding models) and multiplied by a per-model price:

* Built-in prices: `text-embedding-3-small` **$0.02/1M**, `text-embedding-3-large`
  **$0.13/1M**, `text-embedding-ada-002` **$0.10/1M**.
* Local / self-hosted models default to **$0.00** (free).
* Override any model with `EMBEDDING_PRICE_PER_1M`.

The 60 tokens embedded in run 1, on paid models:

| Model                   | Price / 1M | Cost for 60 tokens |
| ----------------------- | ---------- | ------------------ |
| nomic-embed-text (local)| $0.0000    | $0.0000000         |
| text-embedding-3-small  | $0.0200    | $0.0000012         |
| text-embedding-ada-002  | $0.1000    | $0.0000060         |
| text-embedding-3-large  | $0.1300    | $0.0000078         |

## How skipping works (Task 4)

Before embedding, the previous store is loaded and every new chunk is matched by
`id` **and** `text`:

* Same `id` + same `text` → **skipped** (cached vector carried into the new store).
* Same `id` but different `text` → **re-embedded** (the source changed; stale
  vectors are never reused).
* New `id` → embedded.

## Configuration

| Variable                     | Default                  | Purpose                          |
| ---------------------------- | ------------------------ | -------------------------------- |
| `EMBEDDING_BATCH_SIZE`       | `64`                     | Chunks per API request           |
| `EMBEDDING_MAX_RETRIES`      | `5`                      | Retries per batch (transient)    |
| `EMBEDDING_RETRY_BASE_DELAY` | `1.0`                    | Backoff base (seconds)           |
| `EMBEDDING_PRICE_PER_1M`     | *(by model)*             | USD per 1M tokens for the estimate |
| `EMBEDDING_MODEL`            | `nomic-embed-text`       | Embedding model                  |
| `EMBEDDING_OUTPUT`           | `../data/embeddings/<corpus>-embeddings.json` | Vector store |

Run it from `backend/`:

```bash
uv run python scripts/embed_corpus.py          # default corpus: cleaned_corpus
uv run python scripts/embed_corpus.py my_corpus
```
