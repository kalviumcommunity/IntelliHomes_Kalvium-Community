"""
history_manager.py — Context-window-aware conversation history for RAG assistants.

Manages a multi-turn message history (system + alternating user/assistant),
measures token counts via tiktoken (with fallback), and applies trim /
summarise strategies to keep requests safely within the model's context window.
"""

import warnings

# ── Token counter (reusable from Concept 5) ──────────────────────────────

_enc = None  # lazy-loaded tiktoken encoding


def _get_encoding():
    """Lazy-load the tiktoken encoding, falling back to a heuristic."""
    global _enc
    if _enc is not None:
        return _enc
    try:
        import tiktoken

        _enc = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        warnings.warn(f"tiktoken unavailable ({exc}); using character heuristic")
        _enc = None
    return _enc


def count_tokens(text: str | None) -> int:
    """Return the number of tokens in *text*.

    Uses OpenAI's ``cl100k_base`` encoding when tiktoken is available
    (cached locally), otherwise falls back to a heuristic:
    ≈1 token per 4 characters for English text.
    """
    if text is None:
        return 0

    enc = _get_encoding()
    if enc is not None:
        return len(enc.encode(text))
    # Simple heuristic: ~4 chars per token for English
    return max(1, len(text) // 4)


# ── History manager ───────────────────────────────────────────────────────


class ChatHistory:
    """Maintain a multi-turn conversation history and keep it under a token budget.

    Parameters
    ----------
    system_prompt : str
        The system-level instruction (always preserved).
    token_budget : int
        Maximum total tokens allowed for the whole message list
        (including system prompt).  Default 6000 — a safe zone for
        8K-context models.
    strategy : {"trim", "summarise"}
        What to do when history exceeds the budget.
    """

    def __init__(
        self,
        system_prompt: str,
        token_budget: int = 6000,
        strategy: str = "trim",
    ) -> None:
        self.token_budget = token_budget
        self.strategy = strategy
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

    # ── Query helpers ─────────────────────────────────────────────────

    def total_tokens(self) -> int:
        """Sum of *content* tokens across every message in *self.messages*."""
        return sum(count_tokens(m["content"]) for m in self.messages)

    def is_over_budget(self) -> bool:
        """Return True if the current history exceeds the token budget."""
        return self.total_tokens() > self.token_budget

    def turn_count(self) -> int:
        """Number of user+assistant message pairs (excluding system)."""
        return (len(self.messages) - 1) // 2

    # ── Mutation helpers ──────────────────────────────────────────────

    def add_message(self, role: str, content: str) -> None:
        """Append a single message (usually ``"user"`` or ``"assistant"``)."""
        self.messages.append({"role": role, "content": content})

    def add_user(self, content: str) -> None:
        """Shorthand: append a user turn."""
        self.add_message("user", content)

    def add_assistant(self, content: str) -> None:
        """Shorthand: append an assistant turn."""
        self.add_message("assistant", content)

    # ── Budget-management strategies ──────────────────────────────────

    def trim(self) -> int:
        """Drop the oldest non-system turns until history fits the budget.

        Returns the number of messages removed (0 if nothing needed).
        """
        removed = 0
        while self.total_tokens() > self.token_budget and len(self.messages) > 2:
            self.messages.pop(1)  # index 1 = oldest non-system message
            removed += 1
        return removed

    def summarise(self, llm_completion) -> int:
        """Replace older turns with a single model-generated summary.

        The caller supplies a function ``llm_completion(messages)`` that
        returns the content of an assistant reply.  This lets the method
        work with any OpenAI-compatible client without importing one here.

        Returns ``0`` if budget was already fine or there was nothing to
        summarise; otherwise returns the number of *original* messages
        that were replaced.
        """
        if not self.is_over_budget():
            return 0
        # Keep system + at least the most recent 2 messages (last user + assistant)
        if len(self.messages) <= 3:
            return 0

        # Everything between system (index 0) and the last 2 messages
        old_turns = self.messages[1:-2]
        if not old_turns:
            return 0

        turns_text = "\n".join(f"{m['role']}: {m['content']}" for m in old_turns)

        summary_prompt = (
            "Summarise the following conversation turns in a concise paragraph. "
            "Preserve key facts, user intents, decisions, and any retrieved context "
            "that is still relevant.\n\n"
            f"{turns_text}"
        )

        summary_content = llm_completion([{"role": "user", "content": summary_prompt}])

        # Replace the old turns with a single synthetic assistant message
        summary_msg = {
            "role": "assistant",
            "content": f"[Summary of earlier conversation]: {summary_content}",
        }

        # Rebuild: [system, summary_msg, ...most recent 2 messages]
        recent = self.messages[-2:]
        self.messages = [self.messages[0], summary_msg] + recent

        return len(old_turns)

    def enforce_budget(self, llm_completion=None) -> dict:
        """Apply the active strategy to keep history under the token budget.

        Parameters
        ----------
        llm_completion : callable or None
            Required only when ``strategy == "summarise"``.  A function that
            accepts a ``list[dict]`` (messages) and returns the content string
            of the assistant reply.

        Returns
        -------
        dict with keys ``action``, ``tokens_before``, ``tokens_after``,
        ``messages_before``, ``messages_after``, and optionally ``removed``.
        """
        before_tok = self.total_tokens()
        before_cnt = len(self.messages)

        result: dict = {
            "action": "none",
            "tokens_before": before_tok,
            "tokens_after": before_tok,
            "messages_before": before_cnt,
            "messages_after": before_cnt,
            "removed": 0,
        }

        if not self.is_over_budget():
            return result

        if self.strategy == "trim":
            removed = self.trim()
            result.update(action="trim", removed=removed)

        elif self.strategy == "summarise":
            if llm_completion is None:
                raise ValueError("llm_completion is required when strategy='summarise'")
            removed = self.summarise(llm_completion)
            result.update(action="summarise", removed=removed)

        result["tokens_after"] = self.total_tokens()
        result["messages_after"] = len(self.messages)
        return result

    # ── Convenience: the full ask() loop ─────────────────────────────

    def ask(
        self,
        user_msg: str,
        llm_completion,
        *,
        retrieve_chunks: list[str] | None = None,
        print_callback=None,
    ) -> str:
        """Full turn: inject user message (and optional chunks), enforce
        budget, call the LLM, store the reply, and return it.

        Parameters
        ----------
        user_msg : str
            The user's current utterance.
        llm_completion : callable
            ``llm_completion(messages)`` → assistant content string.
        retrieve_chunks : list[str] | None
            Optional retrieved document chunks prepended to the user message.
        print_callback : callable or None
            If given, called with a ``dict`` after each step (useful for demo
            output).

        Returns
        -------
        The assistant reply string.
        """
        # Optionally prepend retrieved chunks to the user message
        final_msg = user_msg
        if retrieve_chunks:
            ctx = "\n\n---\n".join(retrieve_chunks)
            final_msg = (
                f"Retrieved context:\n{ctx}\n\n" f"---\n\nUser question: {user_msg}"
            )

        self.add_user(final_msg)

        # Enforce budget
        report = self.enforce_budget(llm_completion)
        if print_callback:
            print_callback(report)

        # Call the LLM
        reply = llm_completion(self.messages)
        self.add_assistant(reply)
        return reply
