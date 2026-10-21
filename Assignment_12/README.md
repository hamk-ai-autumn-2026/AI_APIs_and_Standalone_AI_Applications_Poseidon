# “Lanterns After Rain” — AI Music Generation

This project generates a 120-second instrumental track with Stability AI’s Stable Audio 2.5 hosted on Replicate. The full prompt is in [`prompt.txt`](./prompt.txt), and [`generate_music.py`](./generate_music.py) contains the complete API client and download flow. It uses only Python’s standard library.

## Generate the song

Requirements: Python 3.10+, a Replicate API token, and network access. Replicate currently charges per generated audio file; see the [model page](https://replicate.com/stability-ai/stable-audio-2.5) for current pricing and schema.

In PowerShell, set your token for the current terminal session and run:

```powershell
$env:REPLICATE_API_TOKEN = "your-token"
python .\generate_music.py
```

The script submits the exact text from `prompt.txt` with duration `120`, seed `1042`, 8 steps, and CFG scale 4. It downloads the model’s MP3 output to `generated/lanterns-after-rain.mp3`.

You can change the duration, seed, prompt, or output path:

```powershell
python .\generate_music.py --duration 120 --seed 1042 --output .\generated\my-track.mp3
python .\generate_music.py --prompt-file .\prompt.txt
```

Supported duration is 1–190 seconds. The model returns MP3 audio, so use an `.mp3` output extension.

## Example song

- **Title:** Lanterns After Rain
- **Style:** Instrumental indie-electronic soundtrack
- **Length:** 120 seconds
- **Prompt:** [`prompt.txt`](./prompt.txt)
- **Output after generation:** `generated/lanterns-after-rain.mp3`

No generated audio is included yet: this workspace has no `REPLICATE_API_TOKEN`, so I could not submit a model run or create an authentic model-generated example file here. Run the command above with your token to produce the song.

## Model and API

- [Stable Audio 2.5 on Replicate](https://replicate.com/stability-ai/stable-audio-2.5)
- [Replicate predictions API](https://replicate.com/docs/topics/predictions/create-a-prediction)
