from conversation.conversational_rag import (
    ConversationHistory,
    rewrite_query,
    retrieve_context,
    build_context,
    generate_answer,
    add_turn,
)

import chromadb


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

COLLECTION_NAME = "property_chunks"
N_RESULTS = 3


# ------------------------------------------------------------
# CHROMA
# ------------------------------------------------------------

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name=COLLECTION_NAME
)


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def print_chunks(chunks):
    print("\n===== RETRIEVED CONTEXT =====")

    if not chunks:
        print("No supporting context found.")
        return

    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")

        metadata = chunk.get("metadata", {})

        print(f"Source: {metadata.get('source', 'unknown')}")
        print(f"Section: {metadata.get('section', 'unknown')}")
        print(f"Position: {metadata.get('position', 'unknown')}")

        if "distance" in chunk:
            print(f"Distance: {chunk['distance']}")

        print(f"Text: {chunk.get('text', '')}")


def run_turn(user_question, history):
    print("\n" + "=" * 60)
    print("USER")
    print("=" * 60)
    print(user_question)

    # --------------------------------------------------------
    # 1. Rewrite question using conversation history
    # --------------------------------------------------------

    rewritten = rewrite_query(
        history,
        user_question
    )

    print("\n===== REWRITTEN RETRIEVAL QUERY =====")
    print(rewritten)

    # --------------------------------------------------------
    # 2. Retrieve using rewritten query
    # --------------------------------------------------------

    chunks = retrieve_context(
        rewritten,
        collection,
        n_results=N_RESULTS
    )

    print_chunks(chunks)

    # --------------------------------------------------------
    # 3. Convert retrieved chunks into text context
    # --------------------------------------------------------

    context = build_context(chunks)

    print("\n===== INJECTED CONTEXT =====")
    print(context if context else "No supporting context available.")

    # --------------------------------------------------------
    # 4. Generate grounded answer
    # --------------------------------------------------------

    print("\n===== ANSWER =====")

    answer = generate_answer(
        user_question,
        context
    )

    print(answer)

    # --------------------------------------------------------
    # 5. Save turn to conversation history
    # --------------------------------------------------------

    add_turn(
        history,
        user_question,
        answer
    )

    return rewritten, chunks, answer


# ------------------------------------------------------------
# MULTI-TURN CONVERSATION
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("CONVERSATIONAL RAG TEST")
    print("=" * 60)

    history = ConversationHistory()

    # --------------------------------------------------------
    # TURN 1
    # --------------------------------------------------------

    q1 = "What documents should I verify before buying a property?"

    rewritten1, chunks1, answer1 = run_turn(
        q1,
        history
    )

    # --------------------------------------------------------
    # TURN 2
    # --------------------------------------------------------

    q2 = "Which one proves ownership?"

    rewritten2, chunks2, answer2 = run_turn(
        q2,
        history
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CONVERSATION SUMMARY")
    print("=" * 60)

    print("\n===== TURN 1 =====")
    print(f"User: {q1}")
    print(f"Rewritten query: {rewritten1}")
    print(f"Answer: {answer1}")

    print("\n===== TURN 2 =====")
    print(f"User: {q2}")
    print(f"Rewritten query: {rewritten2}")
    print(f"Answer: {answer2}")

    print("\n===== CONVERSATION HISTORY =====")

    # Handle either a list-like history or an object
    # exposing messages/history.
    if hasattr(history, "messages"):
        messages = history.messages
    elif hasattr(history, "history"):
        messages = history.history
    else:
        messages = history

    for message in messages:
        if isinstance(message, dict):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            print(f"{role}: {content}")
        else:
            print(message)

    print("\n===== TEST COMPLETE =====")