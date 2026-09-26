import os

from .base import DictionaryProvider
from .prompt import PROMPT_TEMPLATE


class AnthropicProvider(DictionaryProvider):
    """Generates dictionary entries using the Anthropic Claude API."""

    display_name = "Anthropic (Claude)"

    def __init__(self, model: str = "claude-sonnet-4-5", api_key: str | None = None):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'anthropic' package is required for this provider. "
                "Install it with: pip install anthropic"
            ) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "No Anthropic API key found. Set the ANTHROPIC_API_KEY "
                "environment variable or pass api_key explicitly."
            )

        self._client = anthropic.Anthropic(api_key=key)
        self._model = model

    def generate(self, word: str) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1000,
                messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(word=word)}],
            )
        except Exception as exc:  # noqa: BLE001 - surface as a plain RuntimeError
            raise RuntimeError(f"Anthropic API request failed: {exc}") from exc

        text_blocks = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        if not text_blocks:
            raise RuntimeError("Anthropic API returned no text content.")
        return "\n".join(text_blocks)
