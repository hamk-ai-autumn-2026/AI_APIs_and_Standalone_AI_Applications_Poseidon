"""Shared prompt used by every LLM-backed provider.

Keeping the prompt in one place means Anthropic, OpenAI, or any future
provider all ask the model for the exact same schema, which is what makes
them interchangeable from the validator's point of view.
"""

PROMPT_TEMPLATE = """You are a lexicographer. Produce a dictionary entry for the word below.

Word: "{word}"

Respond with ONLY a single JSON object (no markdown fences, no commentary,
no extra text before or after it) with exactly this shape:

{{
  "word": "<the original word, unchanged>",
  "definitions": ["<definition 1 in English>", "..."],
  "synonyms": ["<synonym 1, in the SAME language as the word>", "..."],
  "antonyms": ["<antonym 1, in the SAME language as the word>", "..."],
  "examples": ["<example sentence 1, in the SAME language as the word>", "..."]
}}

Rules:
- "definitions" must always contain at least one entry, written in English.
- "synonyms", "antonyms" and "examples" must be written in the same
  language as the input word (they may be empty lists if none exist).
- Every list item must be a non-empty string.
- Do not wrap the JSON in ```json fences or any other text.
"""
