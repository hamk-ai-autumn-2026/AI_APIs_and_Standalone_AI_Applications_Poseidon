# Streaming speech interpreter

This command-line app records a short microphone utterance, transcribes it,
translates the text, and streams synthesized PCM audio to the speakers as it
arrives. It uses OpenAI speech and chat APIs; API/network and audio-device
failures are reported with the stage that failed.

## Setup and run

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY = "your-api-key"
python interpreter.py --source English --target French --seconds 5
```

The app collects five successful runs by default. Press Enter for each
fixed-duration recording; after five successes it prints the requested median
latency summary and a mean/min/max table, and saves the same report to
`latency_report.txt` in the current directory. Change the destination with
`--report-file results/german-test.txt`. Use `--runs 8` to collect eight
successful runs. Type `q` to stop early; it will clearly mark the report
incomplete if it has fewer than the requested number of successful runs. The
report includes capture duration, STT, translation, TTS time-to-first-audio,
post-capture wait before processing, playback duration, end-of-speech-to-first
playback lag, and processing time (STT + translation + TTS time-to-first-audio).
Run timing statistics use actual API/device measurements; offline tests use
mocked services and do not represent live latency.

Use `--source auto` to detect spoken language, or specify `English`, `German`,
or `Finnish`. Choose the translated voice with `--target German` or
`--target Finnish`, for example:

```powershell
python interpreter.py --source English --target German --seconds 5
python interpreter.py --source Finnish --target German --seconds 5
python interpreter.py --source German --target Finnish --seconds 5
```

The microphone input is sent to the configured API provider. The default
models are `whisper-1`, `gpt-4o-mini`, and `gpt-4o-mini-tts`; they can be
overridden with the corresponding command-line options. The generated speech
is streamed in raw PCM chunks rather than saved as a complete audio file.

## Tests

Run the offline suite with:

```powershell
python -m unittest discover -s tests -v
```

Tests use mocked processing services and an emulated microphone, so they do
not require a key, network access, or physical audio devices.
