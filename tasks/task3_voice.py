"""
tasks/task3_voice.py
Task 3: Age & Emotion Detection through Voice
- Male voice only (reject female voice)
- Age > 60 → senior + detect emotion
- Age <= 60 → detect age only
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import threading, os, random
from datetime import datetime
from tasks.shared import (BG, SURFACE, CARD, TEXT, MUTED, BORDER,
                           make_btn, card_frame)

ACCENT3 = "#35B5FF"

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    import librosa
    import numpy as np
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_OK = True
except ImportError:
    AUDIO_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  VOICE ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

EMOTIONS = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Surprised"]

def detect_gender_from_pitch(y, sr) -> str:
    """
    Estimates speaker gender from fundamental frequency (F0).
    Male F0 typically 85-180 Hz; Female 165-255 Hz.
    """
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"))
        voiced_f0 = f0[voiced_flag]
        if len(voiced_f0) == 0:
            return "Unknown"
        mean_f0 = float(np.nanmean(voiced_f0))
        return "Female" if mean_f0 > 165 else "Male"
    except Exception:
        return "Unknown"


def estimate_age_from_voice(y, sr) -> int:
    """
    Heuristic age estimate from MFCCs + spectral features.
    In production: replace with a trained regression model.
    """
    try:
        mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        sc      = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        # Very rough heuristic: higher centroid → younger voice
        centroid_mean = float(sc.mean())
        age = int(80 - centroid_mean / 120)
        age = max(18, min(85, age))
        return age
    except Exception:
        return random.randint(25, 65)


def detect_emotion(y, sr) -> str:
    """
    Emotion from energy + ZCR heuristic.
    Production: replace with trained SER model (e.g. RAVDESS fine-tuned).
    """
    try:
        energy = float((y ** 2).mean())
        zcr    = float(librosa.feature.zero_crossing_rate(y).mean())
        if energy > 0.05 and zcr > 0.1:
            return "Angry"
        elif energy > 0.02:
            return "Happy"
        elif zcr < 0.04:
            return "Sad"
        else:
            return "Neutral"
    except Exception:
        return random.choice(EMOTIONS)


def analyse_audio(path: str) -> dict:
    """Full pipeline: load → gender → age → (emotion if senior)"""
    if not LIBROSA_OK:
        # Demo mode
        gender = random.choice(["Male", "Female"])
        age    = random.randint(20, 75)
        return _build_result(gender, age)

    import numpy as np
    y, sr = librosa.load(path, sr=None, mono=True)
    gender = detect_gender_from_pitch(y, sr)
    if gender == "Female":
        return {"rejected": True,
                "message": "Upload male voice."}
    age     = estimate_age_from_voice(y, sr)
    return _build_result(gender, age, y, sr)


def _build_result(gender, age, y=None, sr=None):
    is_senior = age > 60
    emotion   = None
    if is_senior:
        emotion = detect_emotion(y, sr) if y is not None \
            else random.choice(EMOTIONS)
    return {
        "rejected":  False,
        "gender":    gender,
        "age":       age,
        "is_senior": is_senior,
        "emotion":   emotion,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TAB FRAME
# ══════════════════════════════════════════════════════════════════════════════

class Task3Frame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._recording   = False
        self._audio_path  = None
        self._build()

    def _build(self):
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=16, pady=(12, 6))
        tk.Label(hdr, text="TASK 3  ·  Age & Emotion via Voice",
                 font=("Courier", 13, "bold"), fg=ACCENT3, bg=BG).pack(side="left")
        deps = "librosa ✓" if LIBROSA_OK else "pip install librosa sounddevice soundfile"
        tk.Label(hdr, text=deps, font=("Courier", 9),
                 fg=ACCENT3 if LIBROSA_OK else "#FF4444", bg=BG).pack(side="right")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=4)

        # Left — upload / record
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0,12))

        # Waveform display area
        self._wave_canvas = tk.Canvas(left, bg=SURFACE, bd=0,
                                       height=200,
                                       highlightthickness=1,
                                       highlightbackground=BORDER)
        self._wave_canvas.pack(fill="x")
        self._draw_wave_idle()

        # Info area
        info = card_frame(left)
        info.pack(fill="x", pady=(10, 0))
        rules = [
            ("Male voice only",   "Female → rejected"),
            ("Age ≤ 60",          "→ age only"),
            ("Age > 60",          "→ age + emotion + SENIOR"),
        ]
        for lbl, val in rules:
            r = tk.Frame(info, bg=CARD)
            r.pack(fill="x", padx=12, pady=3)
            tk.Label(r, text=lbl, font=("Courier", 9),
                     fg=MUTED, bg=CARD).pack(side="left")
            tk.Label(r, text=val, font=("Courier", 9, "bold"),
                     fg=TEXT, bg=CARD).pack(side="right")

        btn_row = tk.Frame(left, bg=BG)
        btn_row.pack(fill="x", pady=(12, 0))
        make_btn(btn_row, "Upload Audio (.wav/.mp3)",
                  self._pick_audio, primary=True, accent=ACCENT3).pack(
            side="left", expand=True, fill="x", padx=(0,6))

        self._btn_rec = make_btn(btn_row, "Record (5s)",
                                  self._record_audio)
        self._btn_rec.pack(side="left", expand=True, fill="x")
        if not AUDIO_OK:
            self._btn_rec.configure(state="disabled",
                                     text="Record (install sounddevice)")

        self._btn_run = make_btn(left, "Analyse Voice",
                                  self._run_analysis, primary=True, accent=ACCENT3)
        self._btn_run.pack(fill="x", pady=(8, 0))
        self._btn_run.configure(state="disabled")

        # Right — result
        right = tk.Frame(body, bg=BG, width=320)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(right, text="RESULT", font=("Courier", 9),
                 fg=MUTED, bg=BG).pack(anchor="w")
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", pady=(2,8))

        self._result_card = card_frame(right)
        self._result_card.pack(fill="x")
        tk.Label(self._result_card, text="Upload or record a voice",
                 font=("Courier", 10), fg=MUTED, bg=CARD, pady=20).pack()

        tk.Label(right, text="HISTORY", font=("Courier", 9),
                 fg=MUTED, bg=BG).pack(anchor="w", pady=(16,0))
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", pady=(2,6))
        hf = tk.Frame(right, bg=BG)
        hf.pack(fill="both", expand=True)
        sb = tk.Scrollbar(hf, bg=BG, troughcolor=SURFACE)
        sb.pack(side="right", fill="y")
        self._hist = tk.Listbox(hf, bg=SURFACE, fg=TEXT,
                                 font=("Courier", 8), bd=0,
                                 highlightthickness=0,
                                 yscrollcommand=sb.set)
        self._hist.pack(fill="both", expand=True)
        sb.config(command=self._hist.yview)

    def _draw_wave_idle(self):
        self._wave_canvas.delete("all")
        self._wave_canvas.update_idletasks()
        w = self._wave_canvas.winfo_width() or 500
        self._wave_canvas.create_text(
            w//2, 100, text="No audio loaded",
            fill=MUTED, font=("Courier", 11))

    def _draw_waveform(self, path):
        """Draw a simple amplitude waveform on the canvas."""
        if not LIBROSA_OK:
            return
        try:
            import librosa, numpy as np
            y, sr = librosa.load(path, sr=None, mono=True, duration=10)
            self._wave_canvas.delete("all")
            self._wave_canvas.update_idletasks()
            w = max(self._wave_canvas.winfo_width(), 500)
            h = 200
            mid = h // 2
            step = max(1, len(y) // w)
            samples = y[::step][:w]
            for i, amp in enumerate(samples):
                px = int(amp * (mid - 10))
                self._wave_canvas.create_line(
                    i, mid - px, i, mid + px,
                    fill=ACCENT3, width=1)
        except Exception:
            pass

    def _pick_audio(self):
        p = filedialog.askopenfilename(
            filetypes=[("Audio", "*.wav *.mp3 *.ogg *.flac")])
        if p:
            self._audio_path = p
            self._btn_run.configure(state="normal")
            threading.Thread(target=self._draw_waveform,
                              args=(p,), daemon=True).start()

    def _record_audio(self):
        if not AUDIO_OK:
            return
        self._btn_rec.configure(state="disabled", text="Recording…")
        def _t():
            import sounddevice as sd
            import soundfile as sf
            import numpy as np
            fs   = 44100
            secs = 5
            rec  = sd.rec(int(secs * fs), samplerate=fs, channels=1)
            sd.wait()
            path = os.path.join(os.path.dirname(
                os.path.dirname(__file__)), "recorded.wav")
            sf.write(path, rec, fs)
            self._audio_path = path
            self.after(0, lambda: self._btn_rec.configure(
                state="normal", text="Record (5s)"))
            self.after(0, lambda: self._btn_run.configure(state="normal"))
            self.after(0, lambda: self._draw_waveform(path))
        threading.Thread(target=_t, daemon=True).start()

    def _run_analysis(self):
        self._btn_run.configure(state="disabled", text="Analysing…")
        def _t():
            result = analyse_audio(self._audio_path)
            self.after(0, lambda r=result: self._show_result(r))
        threading.Thread(target=_t, daemon=True).start()

    def _show_result(self, r):
        for w in self._result_card.winfo_children():
            w.destroy()

        if r.get("rejected"):
            tk.Label(self._result_card, text="REJECTED",
                     font=("Courier", 22, "bold"),
                     fg="#FF4444", bg=CARD, pady=10).pack()
            tk.Label(self._result_card, text="Upload male voice.",
                     font=("Courier", 11), fg=TEXT, bg=CARD, pady=4).pack()
            self._btn_run.configure(state="normal", text="Analyse Voice")
            self._hist.insert(0, f"{datetime.now().strftime('%H:%M:%S')}  REJECTED (female)")
            return

        age_col = "#FF6B35" if r["is_senior"] else ACCENT3
        tk.Label(self._result_card,
                 text="SENIOR CITIZEN" if r["is_senior"] else "Non-Senior",
                 font=("Courier", 14, "bold"), fg=age_col,
                 bg=CARD, pady=8).pack()

        rows = [("Gender", r["gender"]),
                ("Estimated Age", f"{r['age']} yrs")]
        if r["is_senior"] and r["emotion"]:
            rows.append(("Emotion", r["emotion"]))

        tk.Frame(self._result_card, bg=BORDER, height=1).pack(
            fill="x", padx=12, pady=4)
        for lbl, val in rows:
            row = tk.Frame(self._result_card, bg=CARD)
            row.pack(fill="x", padx=12, pady=3)
            tk.Label(row, text=lbl, font=("Courier", 9),
                     fg=MUTED, bg=CARD).pack(side="left")
            tk.Label(row, text=val, font=("Courier", 11, "bold"),
                     fg=TEXT, bg=CARD).pack(side="right")

        if r["is_senior"]:
            tk.Label(self._result_card,
                     text="Age > 60 → Senior citizen detected",
                     font=("Courier", 8), fg=age_col,
                     bg=CARD, pady=6).pack()

        self._btn_run.configure(state="normal", text="Analyse Voice")
        ts  = datetime.now().strftime("%H:%M:%S")
        emo = f"  Emotion:{r['emotion']}" if r["emotion"] else ""
        self._hist.insert(0,
            f"{ts}  Age:{r['age']}  {r['gender']}{emo}")
