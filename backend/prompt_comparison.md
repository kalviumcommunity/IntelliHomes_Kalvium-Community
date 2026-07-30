# Prompt Comparison

## Prompt A

### System Prompt

"You are a helpful assistant."

### Observation

The response answered the question but lacked clear formatting and specific instructions. The output could vary in style and length.

---

## Prompt B

### System Prompt

You are IntelliHomes AI.

Role:
- Assist staff with real estate questions.

Scope:
- Answer only property-related questions.

Constraints:
- Maximum 100 words.
- Use bullet points.
- Be professional.
- If unsure, state that you don't have enough information.

### Observation

The response was concise, consistently formatted with bullet points, and stayed within the real estate domain.

---

# Parameter Experiments

## Temperature effect

### temperature_low

- Parameters: {"temperature": 0.0}
- Effect: Stable and factual
- Example output: "1. Title deed\n2. Sale agreement\n3. Property tax receipt"

### temperature_high

- Parameters: {"temperature": 0.9}
- Effect: More varied and creative
- Example output: "Here are three documents worth checking before closing: the title deed, the transfer paperwork, and the seller's proof of ownership."

## max_tokens effect

### max_tokens_short

- Parameters: {"max_tokens": 40}
- Effect: Short, constrained response
- Example output: "1. Title deed\n2. Sale agreement"

## stop effect

### stop_truncated

- Parameters: {"stop": ["\n\n"]}
- Effect: Stops once a paragraph break is reached
- Example output: "1. Title deed\n2. Sale agreement"

## Recommended settings for grounded tasks

For a grounded, factual task, use a low temperature such as 0.0 or 0.2, a modest max_tokens limit such as 60-120 for a short answer, and an optional stop sequence if you want to avoid extra text after a concise bullet list. These settings keep the model focused on the requested facts and reduce hallucinated or overly creative wording.

---

# Selected Prompt

Prompt B

## Reason

Prompt B clearly defines the assistant's role, scope, formatting rules, and fallback behaviour. This results in more reliable, professional, and consistent responses, making it more suitable for the IntelliHomes assistant.