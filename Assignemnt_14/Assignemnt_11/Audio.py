
import asyncio
import base64
import json
import os
import queue
import sys
import threading
import time

try:
    import pyaudio
    import websockets
except ImportError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        f"Missing dependency: {missing}. "
        "Run: python -m pip install -r requirements.txt"
    ) from exc


# OpenAI Realtime API
URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime"

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK_FRAMES = 480
PLAYBACK_FRAME_BYTES = 480
VOICE = "alloy"


SESSION_CONFIGURATION = {
    "type": "realtime",
    "output_modalities": ["audio"],
    "instructions": (
        "You are a friendly, patient spoken language-practice tutor. "
        "Speak clearly and concisely. Ask one question at a time, adapt "
        "to the learner's level, and gently correct mistakes when useful."
    ),
    "audio": {
        "input": {
            "format": {
                "type": "audio/pcm",
                "rate": RATE,
            },
            "transcription": {
                "model": "whisper-1",
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 700,
                "create_response": True,
                "interrupt_response": True,
            },
        },
        "output": {
            "format": {
                "type": "audio/pcm",
                "rate": RATE,
            },
            "voice": VOICE,
        },
    },
}


class AudioPlayer:
    def __init__(self, output_stream):
        self.output_stream = output_stream
        self.chunks = queue.Queue()
        self.generation = 0
        self.lock = threading.Lock()
        self.error = None
        self.thread = threading.Thread(
            target=self._play,
            daemon=True,
        )
        self.thread.start()

    def enqueue(self, audio):
        if not audio:
            return

        with self.lock:
            if self.error is not None:
                raise RuntimeError(
                    "Audio playback failed."
                ) from self.error

            self.chunks.put((self.generation, audio))

    def interrupt(self):
        with self.lock:
            self.generation += 1

            while True:
                try:
                    self.chunks.get_nowait()
                except queue.Empty:
                    break

    def close(self):
        self.chunks.put(None)
        self.thread.join(timeout=5)

        if self.error is not None:
            raise RuntimeError(
                "Audio playback failed."
            ) from self.error

    def _play(self):
        try:
            while True:
                item = self.chunks.get()

                if item is None:
                    return

                generation, audio = item

                for offset in range(
                    0, len(audio), PLAYBACK_FRAME_BYTES
                ):
                    frame = audio[
                        offset:offset + PLAYBACK_FRAME_BYTES
                    ]

                    with self.lock:
                        if generation != self.generation:
                            break

                        self.output_stream.write(
                            frame,
                            exception_on_underflow=False,
                        )

        except Exception as exc:
            self.error = exc


async def send_audio(websocket, input_stream, session_ready):
    # Do not send microphone audio until configuration is accepted.
    await session_ready.wait()

    while True:
        audio = await asyncio.to_thread(
            input_stream.read,
            CHUNK_FRAMES,
            exception_on_overflow=False,
        )

        await websocket.send(
            json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio).decode("ascii"),
            })
        )


async def receive_events(websocket, player, session_ready):
    speech_ended_at = None
    first_audio_received = False
    pending_audio = bytearray()
    configuration_sent = False

    async for message in websocket:
        event = json.loads(message)
        event_type = event.get("type")

        # The server sends session.created when the connection opens.
        if event_type == "session.created":
            if not configuration_sent:
                await websocket.send(
                    json.dumps({
                        "type": "session.update",
                        "session": SESSION_CONFIGURATION,
                    })
                )
                configuration_sent = True

        # The server confirms the new settings with session.updated.
        elif event_type == "session.updated":
            if not session_ready.is_set():
                print(
                    "Session ready. Start speaking; "
                    "press Ctrl+C to exit."
                )
                session_ready.set()

        elif event_type == "input_audio_buffer.speech_started":
            # Stop any old response audio when the user interrupts.
            player.interrupt()
            pending_audio.clear()
            speech_ended_at = None
            print("\n[Speech detected; interrupting playback]")

        elif event_type == "input_audio_buffer.speech_stopped":
            speech_ended_at = time.monotonic()
            first_audio_received = False

        elif event_type == (
            "conversation.item.input_audio_transcription.completed"
        ):
            transcript = event.get("transcript", "").strip()

            if transcript:
                print(f"\nYou: {transcript}")

        elif event_type in (
            "response.output_audio_transcript.done",
            "response.audio_transcript.done",
            "response.audio_transcript.completed",
        ):
            transcript = event.get("transcript", "").strip()

            if transcript:
                print(f"Assistant: {transcript}")

        elif event_type in (
            "response.output_audio.delta",
            "response.audio.delta",
        ):
            if speech_ended_at is not None and not first_audio_received:
                latency = (
                    time.monotonic() - speech_ended_at
                ) * 1000

                print(
                    f"[First audio: {latency:.0f} ms "
                    "after speech end]"
                )
                first_audio_received = True

            pending_audio.extend(
                base64.b64decode(event["delta"])
            )

            while len(pending_audio) >= PLAYBACK_FRAME_BYTES:
                player.enqueue(
                    bytes(pending_audio[:PLAYBACK_FRAME_BYTES])
                )
                del pending_audio[:PLAYBACK_FRAME_BYTES]

        elif event_type in (
            "response.output_audio.done",
            "response.audio.done",
        ):
            if pending_audio:
                player.enqueue(bytes(pending_audio))
                pending_audio.clear()

        elif event_type == "error":
            detail = event.get("error", {})
            message = detail.get(
                "message",
                json.dumps(detail),
            )
            raise RuntimeError(
                f"Realtime API error: {message}"
            )


async def run_connection(api_key, input_stream, player):
    session_ready = asyncio.Event()

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    async with websockets.connect(
        URL,
        additional_headers=headers,
        open_timeout=15,
        ping_interval=20,
        ping_timeout=20,
        max_size=None,
    ) as websocket:
        print("Connected to the Realtime API.")

        tasks = {
            asyncio.create_task(
                send_audio(
                    websocket,
                    input_stream,
                    session_ready,
                )
            ),
            asyncio.create_task(
                receive_events(
                    websocket,
                    player,
                    session_ready,
                )
            ),
        }

        try:
            done, _ = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                exception = task.exception()

                if exception is not None:
                    raise exception

            raise ConnectionError(
                "Realtime connection closed."
            )

        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )


def open_audio_devices():
    audio = pyaudio.PyAudio()
    input_stream = None
    output_stream = None

    try:
        input_stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_FRAMES,
        )

        output_stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            frames_per_buffer=CHUNK_FRAMES,
        )

    except Exception:
        if input_stream is not None:
            input_stream.close()

        if output_stream is not None:
            output_stream.close()

        audio.terminate()
        raise

    return audio, input_stream, output_stream


def is_session_expired(exc):
    message = str(exc).lower()

    return (
        "maximum duration of 60 minutes" in message
        or "maximum session duration" in message
        or (
            "session" in message
            and "maximum duration" in message
        )
    )


async def main():
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is missing. Set it in PowerShell "
            "before running this program."
        )

    audio, input_stream, output_stream = open_audio_devices()
    player = AudioPlayer(output_stream)
    retry_delay = 1

    try:
        while True:
            try:
                await run_connection(
                    api_key,
                    input_stream,
                    player,
                )

            except asyncio.CancelledError:
                raise

            except RuntimeError as exc:
                if is_session_expired(exc):
                    print(
                        "\nThe 60-minute session limit was reached. "
                        "Reconnecting in 2 seconds...",
                        file=sys.stderr,
                    )
                    player.interrupt()
                    await asyncio.sleep(2)
                    retry_delay = 1
                    continue

                # Do not endlessly retry configuration or API errors.
                raise

            except (
                OSError,
                TimeoutError,
                websockets.exceptions.WebSocketException,
            ) as exc:
                response = getattr(exc, "response", None)
                status_code = getattr(
                    response,
                    "status_code",
                    None,
                )

                if status_code in (401, 403):
                    raise RuntimeError(
                        "The API rejected the credentials. "
                        "Check your OPENAI_API_KEY."
                    ) from exc

                print(
                    f"\nConnection failed: {exc}. "
                    f"Reconnecting in {retry_delay} second(s)...",
                    file=sys.stderr,
                )

                player.interrupt()
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)

            else:
                retry_delay = 1

    finally:
        try:
            player.close()
        finally:
            try:
                for stream in (input_stream, output_stream):
                    try:
                        if stream.is_active():
                            stream.stop_stream()
                    finally:
                        stream.close()
            finally:
                audio.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting voice tutor.")
