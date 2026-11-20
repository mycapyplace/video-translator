import os
import queue
import site
import threading

import ctranslate2
import numpy as np
from faster_whisper import WhisperModel

MODEL_SIZE = "large-v3"
LOCK_LANGUAGE = "ko"


def _register_rocm_dlls():
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
                print(f"[Whisper] Loading {MODEL_SIZE} on RX 6700 XT (ROCm float16)...")
                model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
                list(
                    model.transcribe(
                        np.zeros(8000, dtype=np.float32), task="translate"
                    )[0]
                )
                print("[Whisper] GPU acceleration active.")
                return model
            except Exception as exc:
                print(f"[Whisper] GPU init failed ({exc}); falling back to CPU int8.")

        print(f"[Whisper] Loading {MODEL_SIZE} on CPU (int8)...")
        return WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        self.text_queue.put(("", f"Loading Whisper {MODEL_SIZE}..."))
        self.model = self._load_model()
        self.text_queue.put(("", "Listening for system audio..."))

        while not self._stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            orig_segments, info = self.model.transcribe(
                chunk,
                task="transcribe",
                language=LOCK_LANGUAGE,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            original_text = " ".join(seg.text.strip() for seg in orig_segments).strip()

            if not original_text:
                continue

            detected_lang = LOCK_LANGUAGE or info.language

            if detected_lang == "en":
                self.text_queue.put((original_text, original_text))
                continue

            trans_segments, _ = self.model.transcribe(
                chunk,
                task="translate",
                language=detected_lang,
                beam_size=3,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            translated_text = " ".join(
                seg.text.strip() for seg in trans_segments
            ).strip()

            self.text_queue.put((original_text, translated_text))
