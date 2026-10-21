import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


MODEL_ENDPOINT = (
    "https://api.replicate.com/v1/models/"
    "stability-ai/stable-audio-2.5/predictions"
)
MAX_DURATION_SECONDS = 190
PREDICTION_TIMEOUT_SECONDS = 900
POLL_INTERVAL_SECONDS = 3


def request_json(
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
    wait_seconds: int | None = None,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    data = None
    method = "GET"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
    if wait_seconds is not None:
        headers["Prefer"] = f"wait={wait_seconds}"

    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Replicate returned HTTP {error.code}: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"Could not connect to Replicate: {error.reason}") from error


def wait_for_prediction(prediction: dict[str, Any], token: str) -> str:
    deadline = time.monotonic() + PREDICTION_TIMEOUT_SECONDS
    while prediction.get("status") in {"starting", "processing"}:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Prediction {prediction.get('id', '(unknown)')} timed out."
            )

        get_url = prediction.get("urls", {}).get("get")
        if not get_url:
            raise RuntimeError("Replicate did not return a prediction status URL.")

        print(f"Generation status: {prediction['status']}...")
        time.sleep(POLL_INTERVAL_SECONDS)
        prediction = request_json(get_url, token)

    if prediction.get("status") != "succeeded":
        detail = prediction.get("error") or prediction.get("logs") or "No details provided."
        raise RuntimeError(f"Music generation {prediction.get('status')}: {detail}")

    output_url = prediction.get("output")
    if not isinstance(output_url, str):
        raise RuntimeError("The model did not return an audio file URL.")
    return output_url


def download_audio(url: str, destination: Path) -> None:
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise RuntimeError("The model returned an invalid audio download URL.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with urlopen(url, timeout=120) as response:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                shutil.copyfileobj(response, temporary_file)

        if temporary_path.stat().st_size == 0:
            raise RuntimeError("The downloaded audio file is empty.")
        os.replace(temporary_path, destination)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Audio download failed with HTTP {error.code}: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"Could not download the audio file: {error.reason}") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_prompt(path: Path) -> str:
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise RuntimeError(f"Prompt file is empty: {path}")
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an original instrumental track with Stable Audio 2.5."
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path(__file__).with_name("prompt.txt"),
        help="UTF-8 text file containing the music-generation prompt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "generated" / "lanterns-after-rain.mp3",
        help="Path where the generated audio will be saved.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help=f"Track length in seconds (1-{MAX_DURATION_SECONDS}; default: 120).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1042,
        help="Seed for reproducible generation (default: 1042).",
    )
    args = parser.parse_args()

    if not 1 <= args.duration <= MAX_DURATION_SECONDS:
        parser.error(f"--duration must be between 1 and {MAX_DURATION_SECONDS} seconds.")

    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        parser.error(
            "Set the REPLICATE_API_TOKEN environment variable before running."
        )

    prompt = load_prompt(args.prompt_file)
    model_input = {
        "prompt": prompt,
        "duration": args.duration,
        "steps": 8,
        "cfg_scale": 4,
        "seed": args.seed,
    }

    print("Submitting track to Stable Audio 2.5 on Replicate...")
    prediction = request_json(
        MODEL_ENDPOINT,
        token,
        payload={"input": model_input},
        wait_seconds=60,
    )
    output_url = wait_for_prediction(prediction, token)
    download_audio(output_url, args.output)
    print(f"Saved generated audio to: {args.output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
