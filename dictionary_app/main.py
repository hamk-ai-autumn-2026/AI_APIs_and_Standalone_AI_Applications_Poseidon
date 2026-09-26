#!/usr/bin/env python3
"""
Dictionary entry generator.

Asks the user for a word, sends it to a chosen LLM provider, and prints a
validated dictionary entry (definitions, synonyms, antonyms, examples) as
JSON.

Providers are fully interchangeable: swapping "openai" for "gemini" (or
adding a brand-new provider in providers/) never requires touching this
file. See providers/__init__.py for the registry.

Available providers: openai, gemini, anthropic, mock, and auto (dynamic
switching: prefers whichever provider has an API key configured, and falls
back to the next candidate if one fails with a network/auth error).

Usage:
    python main.py                          # interactive prompts
    python main.py --provider mock          # skip the provider prompt
    python main.py --provider openai --model gpt-4o-mini
    python main.py --provider gemini --model gemini-3.7-flash
    python main.py --provider auto          # dynamic switching
"""

import argparse
import json
import sys

from input_validation import InputError, sanitize_word
from providers import available_providers, create_provider
from validator import ValidationError, parse_and_validate

MAX_ATTEMPTS = 3


def prompt_for_provider() -> str:
    options = available_providers()
    print("\nAvailable providers:")
    for key, name in options.items():
        print(f"  {key:10s} - {name}")
    while True:
        choice = input(f"Choose a provider [{'/'.join(options)}]: ").strip().lower()
        if choice in options:
            return choice
        print(f"'{choice}' is not a valid provider. Please pick one of: {', '.join(options)}")


def prompt_for_word() -> str:
    while True:
        raw = input("Enter a word to look up (press Enter to quit): ")
        if not raw.strip():
            print("Goodbye!")
            raise SystemExit(0)
        try:
            return sanitize_word(raw)
        except InputError as exc:
            print(exc)


def generate_valid_entry(provider, word: str) -> dict:
    """Ask the provider for an entry, retrying a few times if validation fails."""
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw_output = provider.generate(word)
        except RuntimeError as exc:
            # Provider/backend failure (network, auth, missing package, ...)
            # -- not worth retrying automatically.
            raise
        try:
            return parse_and_validate(raw_output, expected_word=word)
        except ValidationError as exc:
            last_error = exc
            print(f"  [attempt {attempt}/{MAX_ATTEMPTS}] Invalid model output: {exc}")
    raise ValidationError(
        f"Model output failed validation {MAX_ATTEMPTS} times in a row. "
        f"Last error: {last_error}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a validated dictionary entry using an LLM.")
    parser.add_argument("--provider", choices=list(available_providers()), help="Provider to use (skips the interactive prompt).")
    parser.add_argument("--model", help="Optional model override passed to the provider's constructor.")
    parser.add_argument("--word", help="Word to look up (skips the interactive prompt).")
    args = parser.parse_args()

    provider_key = args.provider or prompt_for_provider()

    try:
        kwargs = {"model": args.model} if args.model else {}
        provider = create_provider(provider_key, **kwargs)
    except (ValueError, RuntimeError) as exc:
        print(f"Could not initialize provider '{provider_key}': {exc}", file=sys.stderr)
        return 1

    if args.word:
        try:
            word = sanitize_word(args.word)
        except InputError as exc:
            print(f"Invalid --word value: {exc}", file=sys.stderr)
            return 1
    else:
        word = prompt_for_word()

    print(f"\nGenerating dictionary entry for '{word}' using {provider.display_name}...")
    try:
        entry = generate_valid_entry(provider, word)
    except (RuntimeError, ValidationError) as exc:
        print(f"Failed to generate a valid dictionary entry: {exc}", file=sys.stderr)
        return 1

    print("\nValid entry:\n")
    print(json.dumps(entry, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
