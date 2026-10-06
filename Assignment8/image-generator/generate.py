import argparse
import base64
import os
import re
import sys
import uuid
from pathlib import Path

from openai import OpenAI


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Command-line image generation utility"
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help="Description of the image to generate"
    )

    parser.add_argument(
        "--aspect-ratio",
        default="1:1",
        choices=["1:1", "16:9", "9:16"],
        help="Image aspect ratio"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of images to generate"
    )

    parser.add_argument(
        "--output",
        default="images",
        help="Directory where generated images will be saved"
    )

    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Things that should not appear in the image"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed, if supported"
    )

    parser.add_argument(
        "--quality",
        choices=["low", "medium", "high"],
        default=None,
        help="Image quality"
    )

    return parser.parse_args()


def validate_arguments(args):
    if not args.prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    if args.count < 1:
        raise ValueError("Count must be at least 1.")

    if args.count > 10:
        raise ValueError("Count cannot be greater than 10.")

    if args.seed is not None and args.seed < 0:
        raise ValueError("Seed cannot be negative.")


def get_image_size(aspect_ratio):
    sizes = {
        "1:1": "1024x1024",
        "16:9": "1536x1024",
        "9:16": "1024x1536",
    }

    return sizes[aspect_ratio]


def create_filename(prompt, output_directory):
    safe_prompt = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        prompt
    )

    safe_prompt = safe_prompt.strip("_")
    safe_prompt = safe_prompt[:50]

    unique_id = uuid.uuid4().hex[:8]

    return output_directory / f"{safe_prompt}_{unique_id}.png"


def generate_image(client, args, image_size):
    prompt = args.prompt

    if args.negative_prompt:
        prompt += (
            "\n\nAvoid the following elements: "
            + args.negative_prompt
        )

    request = {
        "model": "gpt-image-2",
        "prompt": prompt,
        "size": image_size,
    }

    if args.quality:
        request["quality"] = args.quality

    if args.seed is not None:
        print(
            "Warning: seed was provided, "
            "but this configuration does not send it to the API."
        )

    return client.images.generate(**request)


def extract_image_bytes(response):
    if not response.data:
        raise RuntimeError("The API returned no image data.")

    image = response.data[0]

    if getattr(image, "b64_json", None):
        return base64.b64decode(image.b64_json)

    raise RuntimeError(
        "The API did not return Base64 image data."
    )


def main():
    try:
        args = parse_arguments()

        validate_arguments(args)

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            print(
                "Error: OPENAI_API_KEY environment variable "
                "is not set."
            )
            sys.exit(1)

        client = OpenAI(api_key=api_key)

        output_directory = Path(args.output)
        output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        image_size = get_image_size(
            args.aspect_ratio
        )

        print()
        print("Image Generator")
        print("----------------")
        print(f"Prompt: {args.prompt}")
        print(f"Aspect ratio: {args.aspect_ratio}")
        print(f"Count: {args.count}")
        print(f"Output: {output_directory}")
        print()

        saved_files = []

        for number in range(1, args.count + 1):
            print(
                f"Generating image "
                f"{number}/{args.count}..."
            )

            try:
                response = generate_image(
                    client,
                    args,
                    image_size
                )

                image_bytes = extract_image_bytes(
                    response
                )

                filename = create_filename(
                    args.prompt,
                    output_directory
                )

                with open(filename, "wb") as file:
                    file.write(image_bytes)

                saved_files.append(filename)

                print(f"Saved: {filename}")
                print()

            except Exception as error:
                print(
                    f"Error generating image "
                    f"{number}: {error}"
                )

        print("----------------")
        print("Finished.")

        if saved_files:
            print("Saved files:")

            for filename in saved_files:
                print(f"  {filename}")
        else:
            print("No images were generated.")

    except ValueError as error:
        print(f"Invalid argument: {error}")
        sys.exit(1)

    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
