from generation.grounded_generator import generate_answer


# Known supporting chunks from our corpus.
SUPPORTING_CONTEXT = """
Source: property_guide.txt
Section: Section 1
Position: 0
Text: Buying property requires verifying ownership documents.

Source: property_guide.txt
Section: Section 2
Position: 1
Text: A Title Deed confirms legal ownership.

Source: property_guide.txt
Section: Section 3
Position: 2
Text: A Survey Plan defines property boundaries.

Source: taxes.txt
Section: Section 1
Position: 0
Text: Property Tax Receipts confirm that all taxes have been paid.
"""


QUESTION = "What documents should I verify before buying a property?"


print("=" * 60)
print("TASK 1 + 2: GROUNDED GENERATION")
print("=" * 60)

print("\n===== QUESTION =====")
print(QUESTION)

print("\n===== SUPPORTING CONTEXT =====")
print(SUPPORTING_CONTEXT)

answer = generate_answer(
    question=QUESTION,
    context=SUPPORTING_CONTEXT,
)

print("\n===== GROUNDED ANSWER =====")
print(answer)


print("\n" + "=" * 60)
print("TASK 3: MISSING-CONTEXT FALLBACK")
print("=" * 60)

missing_question = "What is the exact market value of this property today?"

fallback = generate_answer(
    question=missing_question,
    context="",
)

print("\n===== QUESTION =====")
print(missing_question)

print("\n===== FALLBACK OUTPUT =====")
print(fallback)


print("\n" + "=" * 60)
print("TASK 4: WITH RETRIEVAL VS WITHOUT RETRIEVAL")
print("=" * 60)

print("\n===== WITH RETRIEVAL =====")

with_retrieval = generate_answer(
    question=QUESTION,
    context=SUPPORTING_CONTEXT,
)

print(with_retrieval)

print("\n===== WITHOUT RETRIEVAL =====")

without_retrieval = generate_answer(
    question=QUESTION,
    context="",
)

print(without_retrieval)
