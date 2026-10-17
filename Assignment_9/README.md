# Sports Shoe Marketing Generator

## Description
A Python application that uses the OpenAI API and GPT-4.1 mini model to analyze sports shoe images and generate product descriptions and marketing slogans.

## Features
- Analyzes sports shoe images.
- Generates product descriptions and marketing slogans.
- Identifies visible product features.
- Separates visible details from unverified claims.
- Saves generated results to a text file.

## Technologies
- Python
- OpenAI API
- GPT-4.1 mini
- `openai` and `python-dotenv` libraries

## Usage

```bash
python product_marketing.py --images images/sport_shoes_front.jpg images/sport_shoes_side.jpg --info "Sports shoes for running and everyday workouts" --slogans 5 --output marketing_report.txt
```

## Output
The application generates a product description, five marketing slogans, visible image details, user-provided facts, and unverified claims. Results can be saved to `marketing_report.txt`.

## Conclusion
This project demonstrates how AI and Python can automate sports shoe marketing content generation using product images.