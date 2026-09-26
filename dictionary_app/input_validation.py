"""
Validates and normalizes what the user types before it's ever sent to a
provider.

Enforces:
  - exactly one word (no internal whitespace)
  - at least one alphabetic character (rejects pure numbers/symbols)
  - strips stray surrounding punctuation/quotes ("VIno!"" -> "vino")
  - normalizes case (lowercase) so lookups are consistent
"""

# Punctuation/quote characters stripped from the outside of the input.
# Anything in the middle of the word (hyphens, apostrophes within a word,
# accented letters, etc.) is left untouched.
_STRIP_CHARS = " \t\r\n'\"!?.,;:()[]{}"


class InputError(ValueError):
    """Raised when the user's input isn't a single valid word."""


def sanitize_word(raw: str) -> str:
    """Clean and validate a raw user input, returning the normalized word.

    Raises InputError with a user-facing message if the input is invalid.
    """
    text = raw.strip(_STRIP_CHARS)

    if not text:
        raise InputError("Please enter a word.")

    # Reject multi-word input (internal whitespace of any kind).
    if len(text.split()) > 1:
        raise InputError("Please enter a single word.")

    # Reject input with no alphabetic characters at all (e.g. "12351234").
    if not any(ch.isalpha() for ch in text):
        raise InputError("Please enter a valid word containing alphabetic letters.")

    return text.lower()
