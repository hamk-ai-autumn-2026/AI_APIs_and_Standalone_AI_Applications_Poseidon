##   Exercise 7 – Image → Text → Image

For this exercise, I created a Python application that converts an image into text and then uses that text to generate a new image.

### How it works

1. I provide an input image.
2. gpt-5.6-luna analyzes the image and creates a detailed description.
3. The description is printed in the terminal.
4. The description is sent to gpt-image-2.
5. A new image is generated and saved as generated_image.png.

### Files

- image_to_text_to_image.py – Python program
- input.jpg – original image
- generated_image.png – generated image

The API key is stored as an environment variable and is not included in the project.