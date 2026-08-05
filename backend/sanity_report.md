# Retrieval Sanity Report

## Results

Tests: 3

Passed: 3

Failed: 0

---

## Top Ranked Sources

| Query | Expected | Retrieved |
|-------|----------|-----------|
| What proves legal ownership? | property_guide.txt | property_guide.txt |
| How do I verify property taxes? | taxes.txt | taxes.txt |
| Who should provide ownership records? | ownership.txt | ownership.txt |

---

## Surprising Case

A query using the phrase "ownership" also matched the Title Deed chunk because both discuss property ownership. This showed that simple text similarity can return semantically related chunks even when they are not the intended source, highlighting the need for embedding-based retrieval in future improvements.