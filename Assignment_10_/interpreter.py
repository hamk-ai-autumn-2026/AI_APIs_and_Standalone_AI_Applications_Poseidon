"""A streaming speech interpreter backed by OpenAI speech and language APIs."""

from __future__ import annotations

import argparse
import io
import os
import statistics
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol


_LANGUAGE_CODES = {
    "english": "en",
    "french": "fr",
    "spanish": "es",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "dutch": "nl",
    "japanese": "ja",
    "korean": "ko",
    "chinese": "zh",
}


class Transcriber(Protocol):
    def transcribe(self, audio: bytes, language: str) -> str: ...


class Translator(Protocol):
    def translate(self, text: str, source_language: str, target_language: str) -> str: ...


class SpeechPlayer(Protocol):
    def speak(self, text: str, language: str, on_first_audio: Callable[[], None]) -> None: ...


@dataclass(frozen=True)
class RunMetrics:
    stt_delay: float
    translation_delay: float
    tts_time_to_first_audio: float
    total_end_of_speech_to_playback: float
    record_delay: float = 0.0
    wait_delay: float = 0.0
    playback_duration: float = 0.0
    processing_delay: float = 0.0


@dataclass(frozen=True)
class Interpretation:
    transcript: str
    translation: str
    metrics: RunMetrics


@dataclass(frozen=True)
class RecordedAudio:
    wav_data: bytes
    recording_started_at: float
    end_of_speech_at: float


class StageError(RuntimeError):
    """A provider or playback error annotated with the stage that failed."""


class Interpreter:
    def __init__(
        self,
        transcriber: Transcriber,
        translator: Translator,
        player: SpeechPlayer,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._transcriber = transcriber
        self._translator = translator
        self._player = player
        self._clock = clock

    def process(
        self,
        audio: bytes,
        source_language: str,
        target_language: str,
        end_of_speech_at: float | None = None,
        recording_started_at: float | None = None,
    ) -> Interpretation:
        if not audio:
            raise ValueError("Recorded audio is empty.")

        processing_started_at = self._clock()
        end_of_speech = self._clock() if end_of_speech_at is None else end_of_speech_at
        stage_start = self._clock()
        try:
            transcript = self._transcriber.transcribe(audio, source_language)
        except StageError:
            raise
        except Exception as exc:
            raise StageError(f"Speech recognition failed: {exc}") from exc
        stt_delay = self._clock() - stage_start
        if not transcript.strip():
            raise StageError("Speech recognition returned an empty transcript.")

        stage_start = self._clock()
        try:
            translation = self._translator.translate(transcript, source_language, target_language)
        except StageError:
            raise
        except Exception as exc:
            raise StageError(f"Translation failed: {exc}") from exc
        translation_delay = self._clock() - stage_start
        if not translation.strip():
            raise StageError("Translation returned empty text.")

        tts_start = self._clock()
        first_audio_at: float | None = None

        def mark_first_audio() -> None:
            nonlocal first_audio_at
            if first_audio_at is None:
                first_audio_at = self._clock()

        try:
            self._player.speak(translation, target_language, mark_first_audio)
        except StageError:
            raise
        except Exception as exc:
            raise StageError(f"Speech generation or playback failed: {exc}") from exc
        if first_audio_at is None:
            raise StageError("Speech generation completed without producing playable audio.")

        return Interpretation(
            transcript=transcript,
            translation=translation,
            metrics=RunMetrics(
                stt_delay=stt_delay,
                translation_delay=translation_delay,
                tts_time_to_first_audio=first_audio_at - tts_start,
                total_end_of_speech_to_playback=first_audio_at - end_of_speech,
                record_delay=(
                    max(0.0, end_of_speech - recording_started_at)
                    if recording_started_at is not None
                    else 0.0
                ),
                wait_delay=max(0.0, processing_started_at - end_of_speech),
                playback_duration=max(0.0, self._clock() - first_audio_at),
                processing_delay=stt_delay + translation_delay + (first_audio_at - tts_start),
            ),
        )


class OpenAITranscriber:
    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def transcribe(self, audio: bytes, language: str) -> str:
        try:
            language_code = _LANGUAGE_CODES.get(language.casefold())
            if language_code is None and len(language) == 2 and language.isascii() and language.isalpha():
                language_code = language.lower()
            options = {"language": language_code} if language_code else {}
            result = self._client.audio.transcriptions.create(
                model=self._model,
                file=("speech.wav", io.BytesIO(audio), "audio/wav"),
                **options,
            )
            return result.text
        except Exception as exc:
            raise StageError(f"Speech recognition API request failed: {exc}") from exc


class OpenAITranslator:
    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        try:
            source = "the detected source language" if source_language.casefold() == "auto" else source_language
            result = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Translate from {source} to {target_language}. "
                            "Return only the translation, preserving the meaning and tone."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
            )
            translated = result.choices[0].message.content
            if not translated:
                raise StageError("Translation API returned no text.")
            return translated
        except StageError:
            raise
        except Exception as exc:
            raise StageError(f"Translation API request failed: {exc}") from exc


class OpenAISpeechPlayer:
    def __init__(self, client: Any, model: str, voice: str, sample_rate: int = 24000) -> None:
        self._client = client
        self._model = model
        self._voice = voice
        self._sample_rate = sample_rate

    def speak(self, text: str, language: str, on_first_audio: Callable[[], None]) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise StageError("Install the audio dependencies with: pip install -r requirements.txt") from exc

        try:
            with sd.RawOutputStream(samplerate=self._sample_rate, channels=1, dtype="int16") as output:
                with self._client.audio.speech.with_streaming_response.create(
                    model=self._model,
                    voice=self._voice,
                    input=text,
                    instructions=f"Speak naturally in {language}.",
                    response_format="pcm",
                ) as response:
                    pending = b""
                    started = False
                    for chunk in response.iter_bytes(chunk_size=4096):
                        pending += chunk
                        even_length = len(pending) - (len(pending) % 2)
                        if even_length:
                            output.write(pending[:even_length])
                            pending = pending[even_length:]
                            if not started:
                                on_first_audio()
                                started = True
                    if pending:
                        raise StageError("TTS returned an incomplete PCM audio sample.")
                    if not started:
                        raise StageError("TTS returned an empty audio stream.")
        except StageError:
            raise
        except Exception as exc:
            raise StageError(f"Streaming TTS or audio playback failed: {exc}") from exc


def record_audio(seconds: float, sample_rate: int = 16000) -> RecordedAudio:
    if seconds <= 0:
        raise ValueError("Recording duration must be greater than zero.")
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise StageError("Install the audio dependencies with: pip install -r requirements.txt") from exc

    print(f"Recording for {seconds:g} seconds...", flush=True)
    try:
        recording_started_at = time.perf_counter()
        recorded = sd.rec(
            int(seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        end_of_speech_at = time.perf_counter()
        frames = recorded.tobytes()
    except Exception as exc:
        raise StageError(f"Microphone recording failed: {exc}") from exc

    wav_data = io.BytesIO()
    with wave.open(wav_data, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)
    return RecordedAudio(
        wav_data=wav_data.getvalue(),
        recording_started_at=recording_started_at,
        end_of_speech_at=end_of_speech_at,
    )


def format_report(runs: Iterable[RunMetrics]) -> str:
    measured = list(runs)
    if not measured:
        raise ValueError("At least one completed run is required for a latency report.")

    durations = [run.total_end_of_speech_to_playback for run in measured]
    lines = [
        f"STT delay: {statistics.median(run.stt_delay for run in measured) * 1000:.1f} ms median",
        (
            "Translation/LLM delay: "
            f"{statistics.median(run.translation_delay for run in measured) * 1000:.1f} ms median"
        ),
        (
            "TTS time-to-first-audio: "
            f"{statistics.median(run.tts_time_to_first_audio for run in measured) * 1000:.1f} ms median"
        ),
        (
            "Total end-of-speech-to-playback delay: "
            f"{statistics.median(durations) * 1000:.1f} ms median"
        ),
        f"Fastest run: {min(durations) * 1000:.1f} ms",
        f"Median run: {statistics.median(durations) * 1000:.1f} ms",
        f"Slowest run: {max(durations) * 1000:.1f} ms",
    ]
    return "\n".join(lines)


def format_session_report(runs: Iterable[RunMetrics]) -> str:
    """Format repeat-run measurements as the mean/min/max table."""
    measured = list(runs)
    if not measured:
        return "No successful runs to summarize."

    stages = (
        ("record", lambda run: run.record_delay),
        ("stt", lambda run: run.stt_delay),
        ("translate", lambda run: run.translation_delay),
        ("tts", lambda run: run.tts_time_to_first_audio),
        ("wait", lambda run: run.wait_delay),
        ("playback", lambda run: run.playback_duration),
        ("lag", lambda run: run.total_end_of_speech_to_playback),
        ("processing", lambda run: run.processing_delay),
    )
    lines = [
        f"Latency over {len(measured)} successful run(s) (ms):",
        f"{'stage':<12}{'mean':>10}{'min':>10}{'max':>10}",
    ]
    for name, get_value in stages:
        values = [get_value(run) * 1000 for run in measured]
        lines.append(
            f"{name:<12}{statistics.mean(values):>10.0f}"
            f"{min(values):>10.0f}{max(values):>10.0f}"
        )
    return "\n".join(lines)


def build_report(
    runs: Iterable[RunMetrics],
    source_language: str,
    target_language: str,
    requested_runs: int,
) -> str:
    measured = list(runs)
    status = "COMPLETE" if len(measured) >= requested_runs else "INCOMPLETE"
    sections = [
        "Speech interpreter latency report",
        f"Run status: {status} ({len(measured)}/{requested_runs} successful runs)",
        f"Languages: {source_language} -> {target_language}",
    ]
    if measured:
        sections.extend(
            [
                "",
                "Latency summary (median unless marked otherwise):",
                format_report(measured),
                "",
                format_session_report(measured),
            ]
        )
    else:
        sections.extend(["", "No successful runs; latency statistics are unavailable."])
    return "\n".join(sections) + "\n"


def save_report(
    report_path: str | Path,
    runs: Iterable[RunMetrics],
    source_language: str,
    target_language: str,
    requested_runs: int,
) -> Path:
    path = Path(report_path)
    path.write_text(
        build_report(runs, source_language, target_language, requested_runs),
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Translate speech and play it in another language.")
    languages = ("auto", "English", "German", "Finnish", "French", "Spanish", "Italian")
    parser.add_argument(
        "--source",
        default="auto",
        choices=languages,
        help="Spoken language, or auto to detect it (default: auto)",
    )
    parser.add_argument(
        "--target",
        default="German",
        choices=languages[1:],
        help="Language to speak in (default: German)",
    )
    parser.add_argument("--seconds", type=float, default=5.0, help="Microphone recording duration")
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of successful recordings to measure before reporting (default: 5)",
    )
    parser.add_argument(
        "--report-file",
        default="latency_report.txt",
        help="Path where the latency report will be saved (default: latency_report.txt)",
    )
    parser.add_argument("--stt-model", default="whisper-1")
    parser.add_argument("--translation-model", default="gpt-4o-mini")
    parser.add_argument("--tts-model", default="gpt-4o-mini-tts")
    parser.add_argument("--voice", default="alloy")
    args = parser.parse_args(argv)

    if args.runs < 1:
        parser.error("--runs must be at least 1.")
    if args.seconds <= 0:
        parser.error("--seconds must be greater than 0.")

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: set the OPENAI_API_KEY environment variable.", file=sys.stderr)
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print("Error: install dependencies with: pip install -r requirements.txt", file=sys.stderr)
        return 2

    client = OpenAI()
    interpreter = Interpreter(
        OpenAITranscriber(client, args.stt_model),
        OpenAITranslator(client, args.translation_model),
        OpenAISpeechPlayer(client, args.tts_model, args.voice),
    )
    metrics: list[RunMetrics] = []
    print(f"Voice interpreter: {args.source} -> {args.target}")
    print(f"Record duration: {args.seconds:g} seconds; collecting {args.runs} successful runs.")

    try:
        while len(metrics) < args.runs:
            try:
                command = input(
                    f"Run {len(metrics) + 1}/{args.runs}: Enter = record"
                    " | q = stop early: "
                ).strip().casefold()
            except EOFError:
                break
            if command == "q":
                break
            if command:
                print("Press Enter to record, or q to quit.")
                continue

            try:
                audio = record_audio(args.seconds)
                result = interpreter.process(
                    audio.wav_data,
                    args.source,
                    args.target,
                    recording_started_at=audio.recording_started_at,
                    end_of_speech_at=audio.end_of_speech_at,
                )
            except (StageError, ValueError) as exc:
                print(f"Run failed: {exc}", file=sys.stderr)
                continue

            metrics.append(result.metrics)
            run_number = len(metrics)
            print(f"[{run_number}] Heard: {result.transcript}")
            print(f"[{run_number}] {args.target}: {result.translation}")
            print(
                f"[{run_number}] Latency: stt {result.metrics.stt_delay * 1000:.0f} | "
                f"translate {result.metrics.translation_delay * 1000:.0f} | "
                f"tts {result.metrics.tts_time_to_first_audio * 1000:.0f} | "
                f"lag {result.metrics.total_end_of_speech_to_playback * 1000:.0f} ms"
            )
    except KeyboardInterrupt:
        print("\nStopping; preparing report.")

    report = build_report(metrics, args.source, args.target, args.runs)
    print()
    print(report, end="")
    try:
        saved_path = save_report(
            args.report_file,
            metrics,
            args.source,
            args.target,
            args.runs,
        )
    except OSError as exc:
        print(f"Could not save report to {args.report_file}: {exc}", file=sys.stderr)
        return 1
    print(f"\nReport saved to: {saved_path.resolve()}")
    return 0 if len(metrics) >= args.runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
