import io
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from interpreter import (
    Interpreter,
    RunMetrics,
    StageError,
    build_report,
    format_report,
    format_session_report,
    record_audio,
    save_report,
)


class FakeTranscriber:
    def __init__(self, result="Hello there.", error=None):
        self.result = result
        self.error = error

    def transcribe(self, audio, language):
        if self.error:
            raise self.error
        return self.result


class FakeTranslator:
    def __init__(self, result="Bonjour.", error=None):
        self.result = result
        self.error = error

    def translate(self, text, source_language, target_language):
        if self.error:
            raise self.error
        return self.result


class FakePlayer:
    def __init__(self, error=None, send_audio=True):
        self.error = error
        self.send_audio = send_audio
        self.spoken = []

    def speak(self, text, language, on_first_audio):
        self.spoken.append((text, language))
        if self.error:
            raise self.error
        if self.send_audio:
            on_first_audio()


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class TimedTranscriber(FakeTranscriber):
    def __init__(self, clock, delay):
        super().__init__()
        self.clock = clock
        self.delay = delay

    def transcribe(self, audio, language):
        self.clock.advance(self.delay)
        return super().transcribe(audio, language)


class TimedTranslator(FakeTranslator):
    def __init__(self, clock, delay):
        super().__init__()
        self.clock = clock
        self.delay = delay

    def translate(self, text, source_language, target_language):
        self.clock.advance(self.delay)
        return super().translate(text, source_language, target_language)


class TimedPlayer(FakePlayer):
    def __init__(self, clock, delay):
        super().__init__()
        self.clock = clock
        self.delay = delay

    def speak(self, text, language, on_first_audio):
        self.clock.advance(self.delay)
        super().speak(text, language, on_first_audio)


class InterpreterTests(unittest.TestCase):
    def setUp(self):
        self.player = FakePlayer()
        self.interpreter = Interpreter(FakeTranscriber(), FakeTranslator(), self.player)

    def test_process_transcribes_translates_and_plays(self):
        result = self.interpreter.process(b"audio", "English", "French")
        self.assertEqual(result.transcript, "Hello there.")
        self.assertEqual(result.translation, "Bonjour.")
        self.assertEqual(self.player.spoken, [("Bonjour.", "French")])

    def test_process_records_nonnegative_stage_latencies(self):
        result = self.interpreter.process(b"audio", "English", "French")
        self.assertGreaterEqual(result.metrics.stt_delay, 0)
        self.assertGreaterEqual(result.metrics.translation_delay, 0)
        self.assertGreaterEqual(result.metrics.tts_time_to_first_audio, 0)
        self.assertGreaterEqual(result.metrics.total_end_of_speech_to_playback, 0)

    def test_rejects_empty_recording(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            self.interpreter.process(b"", "English", "French")

    def test_reports_transcription_failure_with_stage(self):
        interpreter = Interpreter(FakeTranscriber(error=OSError("offline")), FakeTranslator(), self.player)
        with self.assertRaisesRegex(StageError, "Speech recognition failed: offline"):
            interpreter.process(b"audio", "English", "French")

    def test_rejects_empty_translation(self):
        interpreter = Interpreter(FakeTranscriber(), FakeTranslator(result="  "), self.player)
        with self.assertRaisesRegex(StageError, "empty text"):
            interpreter.process(b"audio", "English", "French")

    def test_rejects_missing_audio_from_tts(self):
        interpreter = Interpreter(FakeTranscriber(), FakeTranslator(), FakePlayer(send_audio=False))
        with self.assertRaisesRegex(StageError, "without producing playable audio"):
            interpreter.process(b"audio", "English", "French")

    def test_records_audio_as_mono_16khz_wav(self):
        import numpy as np

        audio_device = SimpleNamespace(
            rec=Mock(return_value=np.zeros((4, 1), dtype="int16")),
            wait=Mock(),
        )
        with patch.dict("sys.modules", {"sounddevice": audio_device}):
            audio = record_audio(0.00025, sample_rate=16000)
        self.assertGreater(audio.end_of_speech_at, 0)
        with wave.open(io.BytesIO(audio.wav_data), "rb") as recorded:
            self.assertEqual(recorded.getnchannels(), 1)
            self.assertEqual(recorded.getframerate(), 16000)
            self.assertEqual(recorded.getsampwidth(), 2)
            self.assertEqual(recorded.getnframes(), 4)

    def test_formats_latency_report_for_five_runs(self):
        runs = [
            RunMetrics(0.01, 0.02, 0.03, delay)
            for delay in (0.10, 0.20, 0.30, 0.40, 0.50)
        ]
        report = format_report(runs)
        self.assertIn("Fastest run: 100.0 ms", report)
        self.assertIn("Median run: 300.0 ms", report)
        self.assertIn("Slowest run: 500.0 ms", report)

    def test_formats_screenshot_style_mean_min_max_report(self):
        runs = [
            RunMetrics(0.8, 0.6, 1.0, 2.4, 5.0, 0.1, 1.5, 2.4),
            RunMetrics(1.2, 1.0, 2.0, 4.2, 0.3, 0.2, 2.0, 4.2),
        ]
        report = format_session_report(runs)
        self.assertIn("Latency over 2 successful run(s) (ms):", report)
        self.assertIn("stage             mean       min       max", report)
        self.assertIn("stt               1000       800      1200", report)
        self.assertIn("translate          800       600      1000", report)
        self.assertIn("lag               3300      2400      4200", report)

    def test_session_report_handles_no_successful_runs(self):
        self.assertEqual(format_session_report([]), "No successful runs to summarize.")

    def test_build_report_includes_requested_summary_and_run_table(self):
        runs = [
            RunMetrics(0.02, 0.015, 0.06, delay)
            for delay in (0.075, 0.095, 0.115, 0.10, 0.09)
        ]
        report = build_report(runs, "English", "German", 5)
        self.assertIn("Run status: COMPLETE (5/5 successful runs)", report)
        self.assertIn("Languages: English -> German", report)
        self.assertIn("STT delay: 20.0 ms median", report)
        self.assertIn("Fastest run: 75.0 ms", report)
        self.assertIn("Latency over 5 successful run(s) (ms):", report)

    def test_save_report_writes_readable_text_file(self):
        run = RunMetrics(0.02, 0.015, 0.06, 0.095)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "latency_report.txt"
            saved_path = save_report(destination, [run], "Finnish", "German", 1)
            content = destination.read_text(encoding="utf-8")

        self.assertEqual(saved_path, destination)
        self.assertIn("Run status: COMPLETE (1/1 successful runs)", content)
        self.assertIn("Languages: Finnish -> German", content)

    def test_five_timed_runs_report_each_stage_and_total(self):
        results = []
        for tts_delay in (0.04, 0.05, 0.06, 0.07, 0.08):
            clock = FakeClock()
            interpreter = Interpreter(
                TimedTranscriber(clock, 0.02),
                TimedTranslator(clock, 0.015),
                TimedPlayer(clock, tts_delay),
                clock=clock,
            )
            results.append(interpreter.process(b"audio", "English", "French").metrics)

        report = format_report(results)
        print("\nFive-run simulated latency report:\n" + report)
        self.assertIn("STT delay: 20.0 ms median", report)
        self.assertIn("Translation/LLM delay: 15.0 ms median", report)
        self.assertIn("TTS time-to-first-audio: 60.0 ms median", report)
        self.assertIn("Fastest run: 75.0 ms", report)
        self.assertIn("Median run: 95.0 ms", report)
        self.assertIn("Slowest run: 115.0 ms", report)


if __name__ == "__main__":
    unittest.main()
