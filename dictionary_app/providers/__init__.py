"""
Provider registry.

This is the ONLY place that needs to know about concrete provider classes.
`main.py` (and any other consumer) works purely against the abstract
`DictionaryProvider` interface returned by `create_provider`, which is what
lets the user switch providers without changing the rest of the application.

To add a new provider:
  1. Create a new module implementing DictionaryProvider (see base.py).
  2. Register it in PROVIDER_REGISTRY below with a short key.
That's it -- no changes needed anywhere else.
"""

from .base import DictionaryProvider
from .anthropic_provider import AnthropicProvider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider
from .mock_provider import MockProvider

#: Env var each provider needs, used by the "auto" fallback mode.
#: Declared before PROVIDER_REGISTRY/AutoProvider import since auto_provider
#: reads it at call time (deferred import breaks the circularity).
PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "mock": None,
}

from .auto_provider import AutoProvider  # noqa: E402

PROVIDER_REGISTRY = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "mock": MockProvider,
    "auto": AutoProvider,
}


def available_providers() -> dict:
    """Return {key: display_name} for every registered provider."""
    return {key: cls.display_name for key, cls in PROVIDER_REGISTRY.items()}


def create_provider(key: str, **kwargs) -> DictionaryProvider:
    """Instantiate a provider by its registry key.

    Any extra kwargs (e.g. model="...", api_key="...") are forwarded to the
    provider's constructor.
    """
    key = key.strip().lower()
    if key not in PROVIDER_REGISTRY:
        valid = ", ".join(sorted(PROVIDER_REGISTRY))
        raise ValueError(f"Unknown provider '{key}'. Valid options: {valid}")
    return PROVIDER_REGISTRY[key](**kwargs)
