"""Multi-model chat using the Groq and OpenRouter free API tiers.

Run:
    pip install streamlit openai
    export GROQ_API_KEY=...
    export OPENROUTER_API_KEY=...
    streamlit run app.py
"""
import logging
import os
from typing import Any, cast

import streamlit as st

# ---------------------------------------------------------------------------
# Model registry: label -> (provider, model id).
# ---------------------------------------------------------------------------
MODELS = {
    "Groq: GPT-OSS 120B": ("groq", "openai/gpt-oss-120b"),
    "Groq: Qwen 3.8 27B": ("groq", "qwen/qwen3.8-27b"),
    "OpenRouter: Apodex 1.1 Mini (free)": (
        "openrouter",
        "apodex/apodex-1.1-mini:free",
    ),
}
PROVIDERS = {
    "groq": {
        "key_name": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "openrouter": {
        "key_name": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
    },
}


def get_api_key(provider: str) -> str | None:
    """Read the key from the environment or st.secrets (never hard-code it)."""
    name = PROVIDERS[provider]["key_name"]
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets file present
        return None


def stream_reply(provider: str, model: str, messages: list[dict]):
    """Yield the reply text chunk by chunk. May raise on API errors."""
    api_key = get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"{PROVIDERS[provider]['key_name']} is not set.")

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=PROVIDERS[provider]["base_url"],
        default_headers={"Accept-Encoding": "gzip, deflate"},
    )
    try:
        stream = cast(
            Any,
            client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                stream=True,
            ),
        )
        for chunk in cast(Any, stream):
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    finally:
        client.close()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Multi-Model Chat", page_icon="💬")
st.title("💬 Multi-Model Chat")

# Per-user chat history lives in session_state (survives Streamlit reruns).
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role", "content", "model"?}

available_models = {
    label: (provider, model)
    for label, (provider, model) in MODELS.items()
    if get_api_key(provider)
}
missing_keys = sorted(
    {
        PROVIDERS[provider]["key_name"]
        for provider, _ in MODELS.values()
        if not get_api_key(provider)
    }
)

with st.sidebar:
    st.header("Settings")
    if not available_models:
        st.error("Set GROQ_API_KEY and/or OPENROUTER_API_KEY to enable chat.")
        st.stop()
    choice = st.selectbox("Model", list(available_models))
    if missing_keys:
        st.caption(f"Set {' and '.join(missing_keys)} to enable those providers.")
    st.caption("Free-tier models may have rate or availability limits.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

provider, model_id = available_models[choice]

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            st.caption(f"— {msg['model']}")

# Handle a new message
if prompt := st.chat_input("Ask something"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Only role/content go to the API (strip our extra "model" field).
    api_messages = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(stream_reply(provider, model_id, api_messages))
            st.caption(f"— {choice}")
            st.session_state.messages.append(
                {"role": "assistant", "content": reply, "model": choice}
            )
        except Exception as e:
            # Keep the prompt in history so it can be retried or referenced.
            logging.exception("Chat request failed for %s", choice)
            st.error(f"{choice} request failed ({type(e).__name__}): {e}")