import os
import queue
import site
import threading

import ctranslate2
import numpy as np
from faster_whisper import WhisperModel


def _register_rocm_dlls():
    """Register all pip-installed ROCm runtime bin directories on Windows."""
    for base in site.getsitepackages():
        if not os.path.isdir(base):
            continue
        for entry in os.listdir(base):
            if entry.startswith("_rocm"):
                bin_dir = os.path.join(base, entry, "bin")
                if os.path.isdir(bin_dir):
                    os.add_dll_directory(bin_dir)


class WhisperTranslator:
    def __init__(self, audio_queue: queue.Queue, text_queue: queue.Queue):
        self.audio_queue = audio_queue
        self.text_queue = text_queue
        self._stop_event = threading.Event()
        self._thread = None
        self.model = None

    def _load_model(self) -> WhisperModel:
        _register_rocm_dlls()
        if ctranslate2.get_cuda_device_count() > 0:
            try:
                print("[Whisper] Loading large-v3 on RX 6700 XT (ROCm float16)...")
                model = WhisperModel("large-v3", device="cuda", compute_type="float16")
                list(
                    model.transcribe(
                        np.zeros(8000, dtype=np.float32), task="translate"
                    )[0]
                )
                print("[Whisper] GPU acceleration active.")
                return model
            except Exception as exc:
                print(f"[Whisper] GPU init failed ({exc}); falling back to CPU int8.")

        print("[Whisper] Loading large-v3 on CPU (int8)...")
        return WhisperModel("large-v3", device="cpu", compute_type="int8")

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        self.text_queue.put("Loading Whisper large-v3 (first run downloads ~3 GB)...")
        self.model = self._load_model()
        self.text_queue.put("Listening for system audio...")

        while not self._stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            segments, info = self.model.transcribe(
                chunk,
                task="translate",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )

            translated = " ".join(seg.text.strip() for seg in segments).strip()
            if translated:
                lang = info.language.upper()
                self.text_queue.put(f"[{lang} → EN] {translated}")
