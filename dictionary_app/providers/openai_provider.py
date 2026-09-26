import os

from .base import DictionaryProvider
from .prompt import PROMPT_TEMPLATE


class OpenAIProvider(DictionaryProvider):
    """Generates dictionary entries using the OpenAI Chat Completions API."""

    display_name = "OpenAI (GPT)"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'openai' package is required for this provider. "
                "Install it with: pip install openai"
            ) from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "No OpenAI API key found. Set the OPENAI_API_KEY "
                "environment variable or pass api_key explicitly."
            )

        self._client = openai.OpenAI(api_key=key)
        self._model = model

    def generate(self, word: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=1000,
                messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(word=word)}],
            )
        except Exception as exc:  # noqa: BLE001 - surface as a plain RuntimeError
            raise RuntimeError(f"OpenAI API request failed: {exc}") from exc

        choice = response.choices[0] if response.choices else None
        content = getattr(getattr(choice, "message", None), "content", None) if choice else None
        if not content:
            raise RuntimeError("OpenAI API returned no text content.")
        return content
