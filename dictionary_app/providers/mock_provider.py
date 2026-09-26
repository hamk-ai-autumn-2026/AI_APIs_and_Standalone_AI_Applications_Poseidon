import json

from .base import DictionaryProvider

# A tiny built-in "dictionary" so the app is runnable and testable
# with no API key and no network access at all.
_SAMPLE_ENTRIES = {
    "kallis": {
        "word": "kallis",
        "definitions": ["expensive", "dear", "costly"],
        "synonyms": ["hintava", "arvokas", "tyyris"],
        "antonyms": ["halpa", "edullinen"],
        "examples": [
            "Tämä auto on liian kallis.",
            "Hän on minulle hyvin kallis ystävä.",
        ],
    }
}


class MockProvider(DictionaryProvider):
    """Offline stand-in provider: no network, no API key required.

    Useful for demos and for testing the validation/CLI logic in isolation
    from any real LLM backend. Falls back to a generic (intentionally
    minimal) entry for words it doesn't recognize.
    """

    display_name = "Mock (offline, no API key)"

    def generate(self, word: str) -> str:
        key = word.strip().lower()
        if key in _SAMPLE_ENTRIES:
            entry = _SAMPLE_ENTRIES[key]
        else:
            entry = {
                "word": word,
                "definitions": [f"(mock) a definition of '{word}'"],
                "synonyms": [],
                "antonyms": [],
                "examples": [f"(mock) an example sentence using '{word}'."],
            }
        return json.dumps(entry)
