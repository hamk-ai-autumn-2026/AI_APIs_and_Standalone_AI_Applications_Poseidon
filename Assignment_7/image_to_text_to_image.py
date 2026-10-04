import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI()

# Ask the user for the input image
image_path = Path(input("Enter image path: "))

if not image_path.exists():
    print("Error: Image file not found.")
    exit(1)

# Read and encode the image
with open(image_path, "rb") as file:
    image_data = base64.b64encode(file.read()).decode("utf-8")

# Detect image format
extension = image_path.suffix.lower()

mime_types = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

mime_type = mime_types.get(extension)

if mime_type is None:
    print("Error: Unsupported image format.")
    exit(1)

# --------------------------------------------------
# STEP 1: IMAGE -> TEXT
# --------------------------------------------------

print("\nAnalyzing image...\n")

vision_response = client.responses.create(
    model="gpt-5.6-luna",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": """
Describe this image in great detail.

Include:
- main subjects
- people and objects
- environment
- background
- colors
- lighting
- composition
- positions of important objects
- visual style
- atmosphere

Write the description so that it can be directly used as a
prompt for an image-generation model.
""",
                },
                {
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{image_data}",
                },
            ],
        }
    ],
)

description = vision_response.output_text

# Print the description
print("=" * 60)
print("GENERATED IMAGE DESCRIPTION")
print("=" * 60)
print(description)
print("=" * 60)

# --------------------------------------------------
# STEP 2: TEXT -> IMAGE
# --------------------------------------------------

print("\nGenerating new image...\n")

image_response = client.images.generate(
    model="gpt-image-2",
    prompt=description,
)

# Decode generated image
generated_image = base64.b64decode(
    image_response.data[0].b64_json
)

# Save locally
output_path = Path("generated_image.png")

with open(output_path, "wb") as file:
    file.write(generated_image)

print(f"Done! Generated image saved as: {output_path}")