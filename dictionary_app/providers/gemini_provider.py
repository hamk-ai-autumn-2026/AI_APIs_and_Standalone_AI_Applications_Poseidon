import os

from .base import DictionaryProvider
from .prompt import PROMPT_TEMPLATE


class GeminiProvider(DictionaryProvider):
    """Generates dictionary entries using Google's Gemini API (google-genai SDK)."""

    display_name = "Gemini"

    def __init__(self, model: str = "gemini-3.7-flash", api_key: str | None = None):
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'google-genai' package is required for this provider. "
                "Install it with: pip install google-genai"
            ) from exc

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "No Gemini API key found. Set the GEMINI_API_KEY (or GOOGLE_API_KEY) "
                "environment variable or pass api_key explicitly."
            )

        self._client = genai.Client(api_key=key)
        self._model = model

    def generate(self, word: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=PROMPT_TEMPLATE.format(word=word),
            )
        except Exception as exc:  # noqa: BLE001 - surface as a plain RuntimeError
            raise RuntimeError(f"Gemini API request failed: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini API returned no text content.")
        return text
