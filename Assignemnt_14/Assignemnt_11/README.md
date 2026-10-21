# Speech-first language practice tutor

This Python command-line app streams microphone audio to the OpenAI Realtime
API and plays spoken replies. Server-side voice activity detection determines
when a turn is complete. While the tutor is speaking, starting to speak again
interrupts local playback and lets the server handle the new turn.

## Setup (Windows PowerShell)

1. Install Python 3.10 or newer.
2. From this folder, install the dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Set your API key in the current PowerShell session:

   ```powershell
   $env:OPENAI_API_KEY = "your-api-key"
   ```

4. Start the app:

   ```powershell
   python Audio.py
   ```

Use headphones to reduce speaker-to-microphone echo. Press Ctrl+C to exit.
The API key must have access to the Realtime API; audio API usage may incur
charges.

## Evaluation checklist

- **Latency:** Try at least five turns and note the `[First audio: ... ms]`
  measurements. Compare the median in a quiet, stable network environment.
- **Transcription:** Speak ten prepared sentences, including names and
  uncommon words; compare the printed `You:` transcript with what you said.
- **Background noise:** Repeat the same phrases in quiet and with typical room
  noise. Note missed speech and false activations. The VAD threshold is 0.5
  and its end-of-turn silence is 700 ms in `Audio.py`.
- **Turn completion:** Pause briefly mid-sentence, then pause after a complete
  sentence. Check whether the tutor waits through short pauses and responds
  after the final pause.
- **Interruption:** Start speaking while the tutor is talking. Playback is
  queued in short frames and cleared when server VAD reports new speech.
- **Echo:** Compare speaker playback with headphones versus open speakers.
  This app does not provide acoustic echo cancellation, so headphones are
  recommended.
- **Network failure:** Temporarily disconnect the network, confirm that a
  reconnect message appears, then restore it and verify that the app reconnects.
  A reconnect creates a new Realtime session.

Latency measurements include network and model response time after the server
reports speech end. Results vary with network conditions, microphone, and
background noise.
