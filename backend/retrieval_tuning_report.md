# Retrieval relevance tuning

## Test queries

- Query: What proves legal ownership?
  Expected sources: property_guide.txt
- Query: How do I verify property taxes?
  Expected sources: taxes.txt
- Query: Who should provide ownership records?
  Expected sources: ownership.txt

## Compared settings

- k1_min0: top_k=1, min_score=0.0, top1 hit rate=0.67, top-k hit rate=0.67
- k3_min0: top_k=3, min_score=0.0, top1 hit rate=0.67, top-k hit rate=1.00
- k3_min0_2: top_k=3, min_score=0.2, top1 hit rate=0.67, top-k hit rate=1.00

## Best settings

- k3_min0: top_k=3, min_score=0.0
- Reason: it achieved the highest hit rate (1.00) while keeping top-k coverage strong.
