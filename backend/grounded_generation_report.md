# Grounded Generation Report

## Query

What documents should I verify before buying a property?

## Supporting Context

The answer was generated using retrieved chunks from:
- property_guide.txt
- taxes.txt

The chunks contained information about ownership documents, Title Deeds, Survey Plans, and Property Tax Receipts.

## Grounded Answer

The generated answer reflected only information contained in the supporting chunks.

## Missing Context

A question without supporting context returned the fallback:

"I don't have enough information in the provided context to answer confidently."

## With vs Without Retrieval

With retrieval, the answer was grounded in the supplied source material.

Without retrieval, the system returned the missing-context fallback rather than inventing unsupported information.

## Result

Grounded generation successfully uses supplied context and avoids unsupported claims when context is unavailable.
