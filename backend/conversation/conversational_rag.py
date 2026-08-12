import os
from typing import List, Dict

from dotenv import load_dotenv
import chromadb
from openai import OpenAI

load_dotenv()

# ============================================================
# Configuration
# ============================================================

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "property_chunks"


# ============================================================
# Conversation History
# ============================================================

class ConversationHistory:
    def __init__(self):
        self.messages: List[Dict[str, str]] = []

    def add_user_message(self, content: str):
        self.messages.append({
            "role": "user",
            "content": content,
        })

    def add_assistant_message(self, content: str):
        self.messages.append({
            "role": "assistant",
            "content": content,
        })

    def get_messages(self):
        return self.messages

    def clear(self):
        self.messages = []


# ============================================================
# Rewrite Follow-up Query
# ============================================================

def rewrite_query(history, user_question):
    """
    Convert a follow-up question into a standalone retrieval query.
    """

    # Accept either ConversationHistory or a normal list.
    if isinstance(history, ConversationHistory):
        messages = history.get_messages()
    elif isinstance(history, list):
        messages = history
    else:
        messages = []

    # First question does not need previous conversation context.
    if not messages:
        return user_question

    history_text = "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in messages
        if isinstance(message, dict)
        and "role" in message
        and "content" in message
    )

    prompt = f"""
Rewrite the user's latest question into a standalone search query.

Use the conversation history to resolve references such as:
- it
- they
- those documents
- that property
- the same property

Do not answer the question.
Return only the rewritten retrieval query.

Conversation history:
{history_text}

Latest user question:
{user_question}

Standalone retrieval query:
"""

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "llama3.2"),
        messages=[
            {
                "role": "system",
                "content": "You rewrite follow-up questions into standalone retrieval queries.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


# ============================================================
# Embedding
# ============================================================

def embed_query(query):
    """
    Generate a 768-dimensional embedding.

    This must match the embedding model/dimension used
    when the ChromaDB collection was created.
    """

    model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    response = client.embeddings.create(
        model=model,
        input=query,
    )

    embedding = list(response.data[0].embedding)

    if len(embedding) != 768:
        raise ValueError(
            f"Expected 768-dimensional embedding, got {len(embedding)}. "
            f"Check EMBEDDING_MODEL and your ChromaDB collection."
        )

    return embedding


# ============================================================
# Retrieve Context
# ============================================================

def retrieve_context(query, collection=None, n_results=3):
    """
    Retrieve chunks using the rewritten standalone query.
    """

    if collection is None:
        chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        collection = chroma_client.get_collection(
            name=COLLECTION_NAME
        )

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    chunks = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, document in enumerate(documents):

        metadata = (
            metadatas[i]
            if i < len(metadatas)
            else {}
        )

        distance = (
            distances[i]
            if i < len(distances)
            else None
        )

        chunks.append({
            "text": document,
            "metadata": metadata,
            "distance": distance,
        })

    return chunks


# ============================================================
# Build Context
# ============================================================

def build_context(chunks):
    """
    Convert retrieved chunks into context for generation.
    """

    if not chunks:
        return ""

    parts = []

    for i, chunk in enumerate(chunks, 1):

        metadata = chunk.get("metadata", {})

        source = metadata.get(
            "source",
            "unknown"
        )

        section = metadata.get(
            "section",
            "unknown"
        )

        position = metadata.get(
            "position",
            "unknown"
        )

        text = chunk.get(
            "text",
            ""
        )

        parts.append(
            f"[Chunk {i}]\n"
            f"Source: {source}\n"
            f"Section: {section}\n"
            f"Position: {position}\n"
            f"Text: {text}"
        )

    return "\n\n".join(parts)


# ============================================================
# Generate Grounded Answer
# ============================================================

def generate_answer(question, context):
    """
    Generate an answer using only retrieved context.
    """

    if not context.strip():
        return (
            "I don't have enough information in the provided "
            "context to answer this question."
        )

    prompt = f"""
You are IntelliHomes AI.

Answer the user's question using ONLY the provided context.

Rules:
- Do not invent information.
- Do not use outside knowledge.
- If the context does not contain enough information, say:
  "I don't have enough information in the provided context to answer this question."
- Keep the answer concise.
- Mention the source when appropriate.

Retrieved context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "llama3.2"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a grounded real-estate assistant. "
                    "Use only supplied context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


# ============================================================
# Helper: Get Retrieval Query
# ============================================================

def get_retrieval_query(history, user_question):
    """
    Return a standalone query for retrieval.
    """

    return rewrite_query(
        history,
        user_question,
    )


# ============================================================
# Add Conversation Turn
# ============================================================

def add_turn(history, user_question, assistant_answer):
    """
    Store both sides of a conversation turn.
    """

    if isinstance(history, ConversationHistory):

        history.add_user_message(
            user_question
        )

        history.add_assistant_message(
            assistant_answer
        )

    elif isinstance(history, list):

        history.append({
            "role": "user",
            "content": user_question,
        })

        history.append({
            "role": "assistant",
            "content": assistant_answer,
        })

    return history