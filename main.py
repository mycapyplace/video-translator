import queue
import tkinter as tk

from transcribe import LoopbackAudioChunker
from translate import WhisperTranslator


class SubtitleOverlay:
    def __init__(self, text_queue: queue.Queue, on_close):
        self.text_queue = text_queue
        self.on_close = on_close
        self._drag_x = 0
        self._drag_y = 0

        self.root = tk.Tk()
        self.root.title("Live English Subtitles")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.88)
        self.root.configure(bg="#111317")

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(920, int(screen_w * 0.72))
        win_h = 96
        pos_x = (screen_w - win_w) // 2
        pos_y = screen_h - win_h - 70
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")

        self.label = tk.Label(
            self.root,
            text="Starting up...",
            fg="#F8FAFC",
            bg="#111317",
            font=("Segoe UI", 15, "bold"),
            wraplength=win_w - 40,
            justify="center",
            padx=20,
            pady=14,
        )
        self.label.pack(expand=True, fill="both")

        # Left-click drag to move; Right-click or Esc to close
        for widget in (self.root, self.label):
            widget.bind("<ButtonPress-1>", self._start_drag)
            widget.bind("<B1-Motion>", self._do_drag)
            widget.bind("<Button-3>", lambda _: self.quit())
        self.root.bind("<Escape>", lambda _: self.quit())

    def _start_drag(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _do_drag(self, event):
        x = self.root.winfo_x() + (event.x - self._drag_x)
        y = self.root.winfo_y() + (event.y - self._drag_y)
        self.root.geometry(f"+{x}+{y}")

    def _poll_captions(self):
        try:
            while True:
                latest_text = self.text_queue.get_nowait()
                self.label.config(text=latest_text)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_captions)

    def run(self):
        self._poll_captions()
        self.root.mainloop()

    def quit(self):
        self.on_close()
        self.root.destroy()


def main():
    audio_q = queue.Queue(maxsize=4)
    text_q = queue.Queue()

    translator = WhisperTranslator(audio_q, text_q)
    chunker = LoopbackAudioChunker(audio_q)

    def shutdown():
        chunker.stop()
        translator.stop()

    overlay = SubtitleOverlay(text_q, on_close=shutdown)
    translator.start()
    chunker.start()
    overlay.run()


if __name__ == "__main__":
    main()
