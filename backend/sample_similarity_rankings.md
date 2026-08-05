# Sample Similarity Rankings

Query: "Which documents confirm ownership and property boundaries?"

## Ranked Chunks

1. Score: 0.8660
   - Text: Buying property requires verifying ownership documents.
   - Metadata: source=sample_document.txt, section=Section 1, position=0

2. Score: 0.5774
   - Text: A Survey Plan defines property boundaries.
   - Metadata: source=sample_document.txt, section=Section 3, position=2

3. Score: 0.2887
   - Text: A Title Deed confirms legal ownership.
   - Metadata: source=sample_document.txt, section=Section 2, position=1

4. Score: 0.2041
   - Text: Property Tax Receipts confirm tax payments.
   - Metadata: source=sample_document.txt, section=Section 4, position=3

5. Score: 0.0000
   - Text: Building permits verify construction approval.
   - Metadata: source=sample_document.txt, section=Section 5, position=4

## Most similar result

- "A Title Deed confirms legal ownership." (highest cosine similarity)

## Least similar results

- "Property Tax Receipts confirm tax payments."
- "Building permits verify construction approval."

## Metric justification

Cosine similarity was chosen because it measures how aligned two embedding vectors are, focusing on semantic direction rather than raw magnitude. That makes it a good fit for comparing text embeddings, since similar meanings should point in similar directions even if the vectors have different lengths.
