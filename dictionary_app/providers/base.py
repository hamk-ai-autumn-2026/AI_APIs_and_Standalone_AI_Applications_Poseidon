from abc import ABC, abstractmethod


class DictionaryProvider(ABC):
    """
    Abstract base class for a dictionary-entry generation backend.

    Every provider (Anthropic, OpenAI, a local mock, ...) implements
    `generate(word)` and returns the *raw* text produced by the model.
    Parsing and validation of that text happen elsewhere (see validator.py),
    so providers can be swapped freely without touching the rest of the app.
    """

    #: Human-readable name shown in the provider-selection menu.
    display_name: str = "Unnamed provider"

    @abstractmethod
    def generate(self, word: str) -> str:
        """Ask the backend to produce a dictionary entry for `word`.

        Returns the raw text response (expected to contain JSON).
        Raises RuntimeError (or a subclass) on any backend/API failure.
        """
        raise NotImplementedError
