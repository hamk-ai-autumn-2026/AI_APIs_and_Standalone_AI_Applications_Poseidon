"""
Validation of model output against the required dictionary-entry schema:

{
  "word": str,
  "definitions": [str, ...]   # must contain at least one entry
  "synonyms":    [str, ...]   # may be empty
  "antonyms":    [str, ...]   # may be empty
  "examples":    [str, ...]   # may be empty
}
"""

import json

REQUIRED_LIST_FIELDS = ("definitions", "synonyms", "antonyms", "examples")


class ValidationError(Exception):
    """Raised when model output does not conform to the required schema."""


def extract_json(raw_text: str) -> dict:
    """Pull a JSON object out of raw model output.

    Handles the common case of models wrapping their answer in ```json
    fences despite being asked not to, and stray leading/trailing text.
    Raises ValidationError if no valid JSON object can be found.
    """
    text = raw_text.strip()

    # Strip ```json ... ``` or ``` ... ``` fences if present.
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # If there's still leading/trailing noise, isolate the outermost {...}.
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Model output is not valid JSON: {exc}") from exc


def validate_entry(entry: dict, expected_word: str | None = None) -> None:
    """Validate a parsed dictionary entry. Raises ValidationError if invalid.

    Rules enforced:
      - `entry` must be a JSON object (dict).
      - "word" must be present and a non-empty string.
      - "definitions", "synonyms", "antonyms", "examples" must all be
        present and be lists of non-empty strings.
      - "definitions" must contain at least one item (an EMPTY definitions
        list is explicitly invalid, per the application's requirements).
      - If `expected_word` is given, it must match "word" (case-insensitive).
    """
    if not isinstance(entry, dict):
        raise ValidationError(f"Expected a JSON object, got {type(entry).__name__}.")

    if "word" not in entry:
        raise ValidationError("Missing required field: 'word'.")
    if not isinstance(entry["word"], str) or not entry["word"].strip():
        raise ValidationError("Field 'word' must be a non-empty string.")

    for field in REQUIRED_LIST_FIELDS:
        if field not in entry:
            raise ValidationError(f"Missing required field: '{field}'.")
        if not isinstance(entry[field], list):
            raise ValidationError(f"Field '{field}' must be a list.")
        for i, item in enumerate(entry[field]):
            if not isinstance(item, str) or not item.strip():
                raise ValidationError(
                    f"Field '{field}' contains an invalid item at index {i}: "
                    "every item must be a non-empty string."
                )

    # The core requirement: an empty definitions list is invalid.
    if len(entry["definitions"]) == 0:
        raise ValidationError("Field 'definitions' must not be empty.")

    if expected_word is not None:
        if entry["word"].strip().lower() != expected_word.strip().lower():
            raise ValidationError(
                f"Returned word '{entry['word']}' does not match requested "
                f"word '{expected_word}'."
            )


def parse_and_validate(raw_text: str, expected_word: str | None = None) -> dict:
    """Convenience wrapper: extract JSON from raw text, then validate it."""
    entry = extract_json(raw_text)
    validate_entry(entry, expected_word=expected_word)
    return entry
