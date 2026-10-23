# Speech Interpreter Latency Report

## Run details

- Source: terminal results in the user-provided screenshot
- Successful runs: 8
- Language pair: Finnish to German
- Units: milliseconds
- Stage latency values are the means from the screenshot's summary table.
- Median end-of-speech-to-playback latency is calculated from the eight
  per-run `lag` values displayed in the screenshot.

## Results

```text
STT delay: 851 ms
Translation/LLM delay: 913 ms
TTS time-to-first-audio: 1,560 ms
Total end-of-speech-to-playback delay: 7,501 ms
Fastest run: 4,461 ms
Median run: 8,536 ms
Slowest run: 10,386 ms
```

## Conclusion

The largest average processing stage is TTS at 1,560 ms. However, playback
(3,764 ms mean) and wait time (3,638 ms mean) are larger contributors to the
overall lag in the screenshot. Investigate what the interpreter measures as
playback and wait before optimizing only the API stages. Streaming playback is
already enabled; shorter utterances or earlier processing of partial speech
may reduce the time before translated audio starts.

These are the eight existing live runs in the screenshot, not a new benchmark
performed while generating this report. The application can save future
measurements from a completed five-run session to `latency_report.txt`.
