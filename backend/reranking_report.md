# Retrieval Re-Ranking Report

## Query

What documents should I verify before buying a property?

## Candidate Retrieval

The system retrieves 10 candidate chunks before selecting the final results.

## Re-Ranking

The 10 candidates are re-ranked using a custom keyword-overlap scoring method.

## Final Selection

The top 3 candidates after re-ranking are selected as the final retrieval context.

## Before and After

The initial ordering is based on vector similarity scores. The final ordering is based on the second-stage re-ranking score.

## Observation

Re-ranking provides a second relevance check and prioritises chunks containing terms directly related to the user's question.

## Sample Output

The complete candidate set, vector scores, re-rank scores, metadata, source text, and final top 3 results are saved in `reranking_output.txt`.
