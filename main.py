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
        self.min_h = 110

        self.root = tk.Tk()
        self.root.title("")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.88)
        self.root.configure(bg="#111317")

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self.win_w = min(1080, int(screen_w * 0.80))

        label_style = {
            "fg": "#F8FAFC",
            "bg": "#111317",
            "font": ("Segoe UI", 20, "bold"),
            "wraplength": self.win_w - 48,
            "justify": "center",
            "padx": 24,
        }

        self.orig_label, self.trans_label = (
            tk.Label(self.root, **label_style) for _ in range(2)
        )
        self.orig_label.pack(fill="x", pady=(12, 4))
        self.trans_label.pack(fill="x", pady=(4, 14))

        self.root.update_idletasks()
        init_h = max(self.min_h, self.root.winfo_reqheight())
        pos_x = (screen_w - self.win_w) // 2
        pos_y = screen_h - init_h - 70
        self.root.geometry(f"{self.win_w}x{init_h}+{pos_x}+{pos_y}")

        for widget in (self.root, self.orig_label, self.trans_label):
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

    def _resize_to_fit(self):
        bottom_y = self.root.winfo_y() + self.root.winfo_height()
        curr_x = self.root.winfo_x()

        self.root.update_idletasks()
        new_h = max(self.min_h, self.root.winfo_reqheight())
        new_y = max(20, bottom_y - new_h)
        self.root.geometry(f"{self.win_w}x{new_h}+{curr_x}+{new_y}")

    def _poll_captions(self):
        updated = False
        try:
            while True:
                orig_text, trans_text = self.text_queue.get_nowait()
                self.orig_label.config(text=orig_text)
                self.trans_label.config(text=trans_text)
                updated = True
        except queue.Empty:
            pass

        if updated:
            self._resize_to_fit()

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
