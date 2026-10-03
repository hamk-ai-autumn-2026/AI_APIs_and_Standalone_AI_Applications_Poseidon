"""AI image generator with a Streamlit web interface (OpenAI Images API).

Run:
    pip install streamlit openai
    export OPENAI_API_KEY=...        # or put it in .streamlit/secrets.toml
    streamlit run image_app.py
"""
import base64
import os
import time

import streamlit as st
from openai import (
    AuthenticationError,
    BadRequestError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

# Each model supports different options, so the UI adapts to the selection.
MODELS = {
    "gpt-image-1": {
        "sizes": {
            "Square (1024×1024)": "1024x1024",
            "Landscape (1536×1024)": "1536x1024",
            "Portrait (1024×1536)": "1024x1536",
        },
        "qualities": ["low", "medium", "high"],
        "max_n": 4,
    },
    "dall-e-3": {
        "sizes": {
            "Square (1024×1024)": "1024x1024",
            "Landscape (1792×1024)": "1792x1024",
            "Portrait (1024×1792)": "1024x1792",
        },
        "qualities": ["standard", "hd"],
        "max_n": 1,  # DALL·E 3 only returns one image per request
    },
}


def get_api_key() -> str | None:
    """Read the key from the environment or st.secrets (never hard-code it)."""
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:  # no secrets file present
        return None


def generate_images(model: str, prompt: str, size: str, quality: str, n: int):
    """Call the API and return a list of (png_bytes, revised_prompt) tuples."""
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(
        api_key=api_key,
        default_headers={"Accept-Encoding": "gzip, deflate"},
    )
    kwargs = dict(model=model, prompt=prompt, size=size, quality=quality, n=n)
    if model == "dall-e-3":
        kwargs["response_format"] = "b64_json"  # gpt-image-1 always returns b64

    response = client.images.generate(**kwargs)
    return [
        (base64.b64decode(item.b64_json), getattr(item, "revised_prompt", None))
        for item in response.data
    ]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Image Generator", page_icon="🎨", layout="wide")
st.title("🎨 AI Image Generator")

# Keep results in session_state: clicking a download button reruns the script,
# and without this the images would disappear.
if "images" not in st.session_state:
    st.session_state.images = []  # newest first

with st.sidebar:
    st.header("Options")
    model = st.selectbox("Model", list(MODELS))
    cfg = MODELS[model]
    size_label = st.selectbox("Aspect ratio / size", list(cfg["sizes"]))
    quality = st.selectbox("Quality", cfg["qualities"])
    n = st.slider(
        "Number of images",
        1,
        cfg["max_n"],
        1,
        disabled=cfg["max_n"] == 1,
        help="DALL·E 3 supports one image per request." if cfg["max_n"] == 1 else None,
    )
    st.caption("These APIs have no negative prompt; describe what to avoid in the prompt itself.")
    if st.button("Clear gallery"):
        st.session_state.images = []
        st.rerun()

prompt = st.text_area(
    "Describe the image",
    placeholder="A watercolor painting of a lighthouse at sunrise, soft pastel colors",
    height=100,
)

if st.button("Generate", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        try:
            with st.spinner(f"Generating with {model}..."):
                results = generate_images(
                    model, prompt.strip(), cfg["sizes"][size_label], quality, n
                )
            for png, revised in results:
                st.session_state.images.insert(
                    0,
                    {
                        "id": time.time_ns(),
                        "png": png,
                        "prompt": prompt.strip(),
                        "revised": revised,
                        "model": model,
                    },
                )
        except AuthenticationError:
            st.error("Authentication failed: check that your API key is valid.")
        except RateLimitError:
            st.error("Rate limit or quota exceeded. Wait a moment or check your billing.")
        except BadRequestError as e:
            # Includes content-policy rejections and unsupported parameters.
            st.error(f"The request was rejected: {e}")
        except OpenAIError as e:
            st.error(f"API error: {e}")
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:  # network problems, bad base64, etc.
            st.error(f"Unexpected error: {e}")
            with st.expander("Technical details"):
                st.exception(e)

# Gallery (newest first)
if st.session_state.images:
    st.subheader("Results")
    cols = st.columns(2)
    for i, img in enumerate(st.session_state.images):
        with cols[i % 2]:
            st.image(img["png"], use_container_width=True)
            st.caption(f"**{img['model']}** — {img['prompt']}")
            if img["revised"] and img["revised"] != img["prompt"]:
                st.caption(f"Model rewrote the prompt as: {img['revised']}")
            st.download_button(
                "⬇ Download PNG",
                data=img["png"],
                file_name=f"image_{img['id']}.png",
                mime="image/png",
                key=f"dl_{img['id']}",
            )