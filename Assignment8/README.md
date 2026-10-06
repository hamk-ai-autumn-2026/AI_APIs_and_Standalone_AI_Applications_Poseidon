# Exercise 8 - Image Generator CLI

## Description

This project is a command-line image generator written in Python. It uses the OpenAI image-generation API to create images from a text prompt and save them locally.

## Requirements

* Python 3
* OpenAI Python package
* OpenAI API key

## Installation

```cmd
pip install openai
```

Set the API key in Windows CMD:

```cmd
set OPENAI_API_KEY=YOUR_API_KEY
```

## Usage

```cmd
python generate.py --prompt "A red wooden cottage beside a Finnish lake" --aspect-ratio 16:9 --count 3 --output images
```

## Supported Arguments

* `--prompt` - Image description
* `--aspect-ratio` - Image size ratio
* `--count` - Number of images to generate
* `--output` - Output folder
* `--negative-prompt` - Elements to avoid
* `--seed` - Seed when supported
* `--quality` - Image quality

## Features

* Command-line parameter validation
* OpenAI image generation
* Multiple image generation
* Error handling
* Safe unique filenames
* Local image saving
* Prints saved filenames

## Output

Generated images are saved in the specified output folder.

Example:

```text
images/
├── image_1.png
├── image_2.png
└── image_3.png
```

## Optional Extension

The optional `fal` and `replicate` providers were not implemented. The current version uses OpenAI.
