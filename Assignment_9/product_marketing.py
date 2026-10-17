import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# Always locate .env relative to this Python file.
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"


def get_client():
    # Locate .env beside this Python file
    project_folder = Path(__file__).resolve().parent
    env_file = project_folder / ".env"

    # Load the key from that exact file
    load_dotenv(dotenv_path=env_file, override=True)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    # Show the file location without exposing the key
    print(f"Checking .env file: {env_file}")

    if not api_key:
        raise RuntimeError(
            "API key is missing or empty. Check your .env file."
        )

    print("API key found.")

    return OpenAI(api_key=api_key)


def encode_image(image_path):
    """Convert an image file into a Base64 data URL."""
    path = Path(image_path).expanduser()

    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    mime_type, _ = mimetypes.guess_type(str(path))

    supported_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
    }

    if mime_type not in supported_types:
        raise ValueError(
            f"Unsupported image format: {path.name}. "
            "Use JPG, PNG, WEBP, or GIF."
        )

    with path.open("rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def generate_marketing_content(
    images,
    info="",
    language="English",
    tone="professional",
    slogans=5,
):
    """Analyze product images and generate marketing content."""
    client = get_client()

    prompt = f"""
You are a careful product marketing copywriter.

Analyze the supplied sports shoe images and additional information.

USER-PROVIDED INFORMATION:
{info.strip() if info.strip() else "No additional information supplied."}

Generate:
1. One clear product description.
2. Exactly {slogans} short marketing slogans.
3. Details directly observable in the images.
4. Facts explicitly supplied by the user.
5. Important details that cannot be verified.

Language: {language}
Tone: {tone}

ACCURACY RULES:
- Use the images and user-provided information together.
- Clearly distinguish visible observations from supplied facts.
- User-provided claims are not independently verified.
- Never invent materials, weight, dimensions, grip, durability,
  cushioning performance, certifications, or technical features.
- Do not infer technical properties from appearance alone.
- Avoid unsupported safety, health, or performance claims.
- Keep every slogan consistent with the available evidence.
- State when important details cannot be determined.

Return only valid JSON in this format:
{{
  "product_description": "Description here",
  "marketing_slogans": ["Slogan 1", "Slogan 2"],
  "visible_observations": ["Visible detail"],
  "user_provided_facts": ["Supplied fact"],
  "unverified_details": ["Unverified detail"]
}}
"""

    content = [{"type": "input_text", "text": prompt}]

    for image_path in images:
        image_url = encode_image(image_path)

        content.append({
            "type": "input_text",
            "text": f"Product image: {Path(image_path).name}",
        })
        content.append({
            "type": "input_image",
            "image_url": image_url,
        })

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )

    try:
        result = json.loads(response.output_text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(
            "The AI did not return valid JSON. Please try again."
        ) from exc

    required_keys = [
        "product_description",
        "marketing_slogans",
        "visible_observations",
        "user_provided_facts",
        "unverified_details",
    ]

    if not isinstance(result, dict):
        raise RuntimeError("The AI response is not a JSON object.")

    for key in required_keys:
        if key not in result:
            raise RuntimeError(f"Missing field in AI response: {key}")

    if not isinstance(result["marketing_slogans"], list):
        raise RuntimeError("The slogans must be a list.")

    if len(result["marketing_slogans"]) != slogans:
        raise RuntimeError(
            f"Expected {slogans} slogans, but received "
            f"{len(result['marketing_slogans'])}. Please try again."
        )

    for key in required_keys:
        if key != "marketing_slogans" and not isinstance(result[key], str if key == "product_description" else list):
            raise RuntimeError(f"Invalid data type for field: {key}")

    return result


def print_results(result):
    """Print all generated content in the terminal."""
    print("\n" + "=" * 55)
    print("SPORTS SHOE PRODUCT DESCRIPTION")
    print("=" * 55)
    print(result["product_description"])

    print("\nMARKETING SLOGANS")
    print("-" * 55)

    for slogan in result["marketing_slogans"]:
        print(f"- {slogan}")

    sections = [
        ("VISIBLE IMAGE DETAILS", "visible_observations"),
        ("USER-PROVIDED FACTS", "user_provided_facts"),
        ("UNVERIFIED DETAILS", "unverified_details"),
    ]

    for heading, key in sections:
        print(f"\n{heading}")
        print("-" * 55)

        if result[key]:
            for item in result[key]:
                print(f"- {item}")
        else:
            print("- None reported.")

    print("\nNote: User-provided claims are not independently verified.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate product descriptions and slogans from images."
    )

    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="One or more product image paths.",
    )
    parser.add_argument(
        "--info",
        default="",
        help="Optional product information.",
    )
    parser.add_argument(
        "--language",
        default="English",
        help="Output language.",
    )
    parser.add_argument(
        "--tone",
        default="professional",
        help="Marketing tone.",
    )
    parser.add_argument(
        "--slogans",
        type=int,
        default=5,
        help="Number of slogans to generate (1-20).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output filename.",
    )

    args = parser.parse_args()

    if not 1 <= args.slogans <= 20:
        parser.error("--slogans must be between 1 and 20.")

    # Resolve relative image paths from the project directory.
    images = [
        str((BASE_DIR / Path(p)).resolve())
        if not Path(p).expanduser().is_absolute()
        else str(Path(p).expanduser().resolve())
        for p in args.images
    ]

    try:
        result = generate_marketing_content(
            images=images,
            info=args.info,
            language=args.language,
            tone=args.tone,
            slogans=args.slogans,
        )

        print_results(result)

        if args.output:
            output_path = Path(args.output).expanduser()

            if not output_path.is_absolute():
                output_path = BASE_DIR / output_path

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"\nResults saved to: {output_path}")

    except FileNotFoundError as exc:
        print(f"\nFILE ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"\nINPUT ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"\nCONFIGURATION OR RESPONSE ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(
            f"\nAPI ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
