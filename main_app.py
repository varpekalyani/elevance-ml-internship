"""
Elevance Skills — ML Internship Project
========================================
All 6 tasks in ONE unified application with a tabbed interface.

Task 1 : Long Hair Gender Detection
Task 2 : Senior Citizen Identification (Video / Webcam)
Task 3 : Age & Emotion Detection via Voice
Task 4 : Sign Language Detection
Task 5 : Car Colour Detection
Task 6 : Nationality Detection

Run:
    python main_app.py
"""

import tkinter as tk
from tkinter import ttk
import os, sys

# ── Colour palette (shared across all tabs) ───────────────────────────────────
BG      = "#0D0D0D"
SURFACE = "#1A1A1A"
CARD    = "#222222"
ACCENT  = "#C8FF00"
TEXT    = "#F0F0F0"
MUTED   = "#888888"
BORDER  = "#333333"

# ── Tab accent colours ────────────────────────────────────────────────────────
TAB_COLORS = {
    "Task 1": "#C8FF00",
    "Task 2": "#FF6B35",
    "Task 3": "#35B5FF",
    "Task 4": "#A78BFA",
    "Task 5": "#FBBF24",
    "Task 6": "#F472B6",
}

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORT ALL TASK MODULES
# ══════════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.dirname(__file__))

from tasks.task1_hair_gender   import Task1Frame
from tasks.task2_senior        import Task2Frame
from tasks.task3_voice         import Task3Frame
from tasks.task4_signlang      import Task4Frame
from tasks.task5_car_colour    import Task5Frame
from tasks.task6_nationality   import Task6Frame


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION WINDOW
# ══════════════════════════════════════════════════════════════════════════════
class ElevanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Elevance Skills — ML Internship  ·  All Tasks")
        self.geometry("1200x780")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(1000, 650)

        self._build_header()
        self._build_notebook()
        self._build_footer()

    def _build_header(self):
        hdr = tk.Frame(self, bg=BG, pady=12)
        hdr.pack(fill="x", padx=24)

        tk.Label(hdr, text="ELEVANCE  SKILLS",
                 font=("Courier", 24, "bold"),
                 fg=ACCENT, bg=BG).pack(side="left")

        tk.Label(hdr, text="  ML Internship — 6 Tasks",
                 font=("Courier", 12), fg=MUTED, bg=BG).pack(side="left")

        tk.Label(hdr, text="Deadline: 22 / 07 / 2026",
                 font=("Courier", 10), fg=MUTED, bg=BG).pack(side="right")

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

    def _build_notebook(self):
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.TNotebook",
                         background=BG, borderwidth=0, tabmargins=0)
        style.configure("Dark.TNotebook.Tab",
                         background=SURFACE, foreground=MUTED,
                         font=("Courier", 10, "bold"),
                         padding=[14, 8], borderwidth=0)
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", CARD)],
                  foreground=[("selected", ACCENT)])

        self._nb = ttk.Notebook(self, style="Dark.TNotebook")
        self._nb.pack(fill="both", expand=True, padx=8, pady=8)

        tasks = [
            ("Task 1 · Hair Gender",      Task1Frame),
            ("Task 2 · Senior Citizen",   Task2Frame),
            ("Task 3 · Voice Age",        Task3Frame),
            ("Task 4 · Sign Language",    Task4Frame),
            ("Task 5 · Car Colour",       Task5Frame),
            ("Task 6 · Nationality",      Task6Frame),
        ]

        for label, FrameClass in tasks:
            frame = FrameClass(self._nb)
            self._nb.add(frame, text=f"  {label}  ")

    def _build_footer(self):
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        foot = tk.Frame(self, bg=BG, pady=6)
        foot.pack(fill="x", padx=24)
        tk.Label(foot,
                 text="training@elevanceskills.com  ·  Built with OpenCV · TensorFlow · Tkinter",
                 font=("Courier", 9), fg=MUTED, bg=BG).pack(side="left")

    def on_close(self):
        # Give each tab a chance to clean up (e.g. stop webcam threads)
        for tab_id in self._nb.tabs():
            frame = self._nb.nametowidget(tab_id)
            if hasattr(frame, "on_close"):
                try:
                    frame.on_close()
                except Exception:
                    pass
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = ElevanceApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
