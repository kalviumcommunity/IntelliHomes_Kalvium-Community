# Chunking Strategy Report

## Strategy Comparison

### Fixed-size with Overlap

- Chunk count: 3
- Average chunk size: 90.67 characters

**Advantages**
- Produces fewer chunks.
- Maintains some context using overlap.

**Disadvantages**
- Can split words or sentences, reducing readability.

---

### Paragraph Chunking

- Chunk count: 5
- Average chunk size: 44.80 characters

**Advantages**
- Preserves complete thoughts and sentence boundaries.
- Easier to understand and retrieve meaningful information.

**Disadvantages**
- Chunk sizes vary depending on the document.

---

## Selected Strategy

Paragraph chunking was selected because IntelliHomes documents are naturally organized into logical sections. Preserving paragraph boundaries keeps related information together, making retrieved results more meaningful and easier to interpret.