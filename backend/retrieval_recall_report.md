# Retrieval recall and precision evaluation

## Labelled query set

- Query: What proves legal ownership?
  Relevant chunk IDs: property_guide-2
- Query: How do I verify property taxes?
  Relevant chunk IDs: taxes-1
- Query: Who should provide ownership records?
  Relevant chunk IDs: ownership-1

## Recall/precision results

- k1: recall@1=0.33, precision@1=0.33
- k3: recall@3=0.67, precision@3=0.22

## Failure analysis

- Best setting: k3 (recall@3=0.67, precision@3=0.22)
- Likely causes for lower-scoring cases: chunking may be too coarse, the query wording may be too broad, or the retrieval signal may need a stronger keyword overlap or metadata filter.
