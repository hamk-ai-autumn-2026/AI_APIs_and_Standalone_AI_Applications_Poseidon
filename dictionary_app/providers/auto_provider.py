import os

from .base import DictionaryProvider

# Real API-backed providers only -- "mock" is deliberately excluded so
# "auto" always reflects genuine API availability/failures.
_DEFAULT_CANDIDATES = ["openai", "gemini", "anthropic"]


class AutoProvider(DictionaryProvider):
    """Meta-provider for dynamic model switching.

    Tries each candidate provider in turn, preferring ones whose API key is
    already set in the environment, and falls back to the next candidate on
    any network/auth/backend error (a RuntimeError from a real provider).
    """

    display_name = "Auto (prefer available key, fall back on error)"

    def __init__(self, candidates: list[str] | None = None, model: str | None = None):
        # Deferred import: avoids a circular import with providers/__init__.py,
        # which imports this class at module load time.
        from . import PROVIDER_REGISTRY, PROVIDER_ENV_VARS

        self._registry = PROVIDER_REGISTRY
        self._model = model
        self._candidates = list(candidates) if candidates else list(_DEFAULT_CANDIDATES)

        # Providers with a configured API key are tried first.
        def has_key(key: str) -> bool:
            env_var = PROVIDER_ENV_VARS.get(key)
            return bool(env_var and os.environ.get(env_var))

        self._candidates.sort(key=lambda key: 0 if has_key(key) else 1)

    def generate(self, word: str) -> str:
        errors = []
        for key in self._candidates:
            try:
                kwargs = {"model": self._model} if self._model else {}
                provider = self._registry[key](**kwargs)
                result = provider.generate(word)
                self.display_name = f"Auto -> {provider.display_name}"
                return result
            except RuntimeError as exc:
                errors.append(f"{key}: {exc}")
                continue
        details = "\n  ".join(errors) if errors else "no candidates configured"
        raise RuntimeError(f"All candidate providers failed:\n  {details}")
