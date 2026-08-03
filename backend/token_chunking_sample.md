# Token-aware chunking sample

## Chosen settings

- Chunk size: 12 tokens
- Overlap: 4 tokens
- Rationale: this keeps chunks small enough for a retrieval pipeline while preserving local context across boundaries. For a typical 8k-context model, a 12-token chunk is intentionally small for demonstration purposes and helps show the overlap effect clearly.

## Example chunks

### Chunk 1

- tokens: 12
- overlap: 4
- text: "A title deed confirms ownership. A survey plan d"

### Chunk 2

- tokens: 12
- overlap: 4
- text: " A survey plan defines the property boundaries."

## Boundary-context example

Without overlap, the phrase "A survey plan defines the property boundaries" may be split awkwardly between chunks. With overlap, the second chunk begins with the tail of the previous sentence, so the boundary remains understandable.
