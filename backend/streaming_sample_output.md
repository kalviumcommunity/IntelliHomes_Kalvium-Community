# Streaming + Citation Sample Interaction

Date: 2026-08-17
Mode: Simulated stream (chunked response for deterministic demo)
Endpoint: POST /chat/stream

## Request

```http
POST /chat/stream
Content-Type: application/json

{"question":"What proves legal ownership of a property?","k":5}
```

## Streamed NDJSON Events (Recorded)

```json
{"type":"status","message":"Retrieving grounded sources..."}
{"type":"citations","citations":[{"marker":"[1]","id":"ownership.txt#0","document":"ownership.txt","section":"Ownership proof","position":"chunk-0","chunk_id":"ownership.txt#0","score":0.88,"text":"A title deed is primary legal ownership proof."}],"hit_count":1}
{"type":"delta","content":"A title deed is primary "}
{"type":"delta","content":"proof of legal ownership"}
{"type":"delta","content":". Verify registration and"}
{"type":"delta","content":" cross-check records [1]."}
{"type":"done","answer":"A title deed is primary proof of legal ownership. Verify registration and cross-check records [1].","citations":[{"marker":"[1]","id":"ownership.txt#0","document":"ownership.txt","section":"Ownership proof","position":"chunk-0","chunk_id":"ownership.txt#0","score":0.88,"text":"A title deed is primary legal ownership proof."}]}
```

## UI Rendering Snapshot (Textual)

- Progressive answer appears token by token while status updates from "Starting stream..." to "Found 1 grounded source(s)."
- Citation panel shown under the answer:
  - [1] ownership.txt - Ownership proof
  - chunk: ownership.txt#0 | position: chunk-0 | score: 0.88
  - Expandable source text:
    - A title deed is primary legal ownership proof.

## Error Case Example

If retrieval/generation fails mid-stream, UI keeps partial answer and shows:

- Stream interrupted.
- Streaming failed: <error details>

The input box remains enabled so the user can retry immediately.
