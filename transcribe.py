import collections
import math
import queue
import threading

import numpy as np
import pyaudiowpatch as pyaudio
from scipy.signal import resample_poly

TARGET_RATE = 16000
FRAME_MS = 30
ENERGY_THRESHOLD = 0.012
PAUSE_SECONDS = 0.40
MIN_SPEECH_SECONDS = 0.65
MAX_CHUNK_SECONDS = 7


class LoopbackAudioChunker:
    def __init__(self, audio_queue: queue.Queue):
        self.audio_queue = audio_queue
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _enqueue_latest(self, chunk: np.ndarray):
        if self.audio_queue.full():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
        self.audio_queue.put_nowait(chunk)

    def _run(self):
        with pyaudio.PyAudio() as p:
            try:
                loopback = p.get_default_wasapi_loopback()
            except Exception as exc:
                print(f"[Audio] No WASAPI loopback output device found: {exc}")
                return

            src_rate = int(loopback["defaultSampleRate"])
            channels = int(loopback["maxInputChannels"])
            frames_per_buffer = int(src_rate * (FRAME_MS / 1000.0))

            gcd = math.gcd(TARGET_RATE, src_rate)
            up = TARGET_RATE // gcd
            down = src_rate // gcd

            print(
                f"[Audio] Capturing loopback: {loopback['name']} ({src_rate} Hz, {channels} ch)"
            )

            stream = p.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=src_rate,
                input=True,
                input_device_index=loopback["index"],
                frames_per_buffer=frames_per_buffer,
            )

            pre_roll = collections.deque(maxlen=int(0.30 / (FRAME_MS / 1000.0)))
            speech_frames = []
            speaking = False
            silence_frames = 0

            max_silence = int(PAUSE_SECONDS / (FRAME_MS / 1000.0))
            min_frames = int(MIN_SPEECH_SECONDS / (FRAME_MS / 1000.0))
            max_frames = int(MAX_CHUNK_SECONDS / (FRAME_MS / 1000.0))

            try:
                while not self._stop_event.is_set():
                    raw = stream.read(frames_per_buffer, exception_on_overflow=False)
                    samples = np.frombuffer(raw, dtype=np.float32)

                    if channels > 1:
                        samples = samples.reshape(-1, channels).mean(axis=1)

                    mono_16k = resample_poly(samples, up, down).astype(np.float32)
                    rms = float(np.sqrt(np.mean(mono_16k**2)))

                    if rms > ENERGY_THRESHOLD:
                        if not speaking:
                            speaking = True
                            speech_frames.extend(pre_roll)
                        speech_frames.append(mono_16k)
                        silence_frames = 0
                    elif speaking:
                        speech_frames.append(mono_16k)
                        silence_frames += 1

                    pre_roll.append(mono_16k)

                    if speaking and (
                        silence_frames >= max_silence
                        or len(speech_frames) >= max_frames
                    ):
                        if len(speech_frames) >= min_frames:
                            self._enqueue_latest(np.concatenate(speech_frames))
                        speech_frames.clear()
                        speaking = False
                        silence_frames = 0
            finally:
                stream.stop_stream()
                stream.close()
