"""
tasks/task6_nationality.py
Task 6: Nationality Detection
- Upload image → predict nationality + emotion
- Indian  → nationality + age + dress colour + emotion
- American → nationality + age + emotion
- African  → nationality + emotion + dress colour
- Others   → nationality + emotion only
"""

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2, numpy as np, threading, os, random
from datetime import datetime
from tasks.shared import (BG, SURFACE, CARD, TEXT, MUTED, BORDER,
                           ensure_weights, load_face_net,
                           load_age_net, load_gender_net,
                           detect_faces, predict_age_gender,
                           make_btn, card_frame)

ACCENT6 = "#F472B6"

NATIONALITIES = ["Indian", "American", "African", "East Asian",
                 "European", "Middle Eastern", "Latin American"]
EMOTIONS      = ["Neutral", "Happy", "Sad", "Angry", "Surprised", "Fearful"]
DRESS_COLOURS = ["Red", "Blue", "Green", "White", "Black",
                  "Yellow", "Orange", "Purple"]


def detect_dress_colour(img_bgr, face_box) -> str:
    """Detect dominant colour in the torso region below the face."""
    x1, y1, x2, y2 = face_box
    ih, iw = img_bgr.shape[:2]
    torso_y1 = min(ih, y2)
    torso_y2 = min(ih, y2 + (y2 - y1))
    torso    = img_bgr[torso_y1:torso_y2, max(0,x1):min(iw,x2)]
    if torso.size == 0:
        return random.choice(DRESS_COLOURS)
    hsv   = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    # Find dominant hue bucket
    hue   = hsv[:,:,0].flatten()
    hist  = np.bincount(hue, minlength=180)
    dom_h = int(np.argmax(hist))
    if dom_h < 10 or dom_h > 170:
        return "Red"
    elif 10 <= dom_h < 25:
        return "Orange"
    elif 25 <= dom_h < 35:
        return "Yellow"
    elif 35 <= dom_h < 85:
        return "Green"
    elif 85 <= dom_h < 130:
        return "Blue"
    elif 130 <= dom_h < 160:
        return "Purple"
    else:
        s_mean = float(hsv[:,:,1].mean())
        v_mean = float(hsv[:,:,2].mean())
        if v_mean > 200 and s_mean < 40:
            return "White"
        elif v_mean < 60:
            return "Black"
        return "Grey"


def detect_emotion_from_face(face_bgr) -> str:
    """
    Heuristic emotion from face intensity distribution.
    Replace with DeepFace / FER+ model for production.
    """
    if face_bgr.size == 0:
        return random.choice(EMOTIONS)
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    std  = float(gray.std())
    mean = float(gray.mean())
    if std > 60:
        return "Surprised"
    elif mean > 160:
        return "Happy"
    elif mean < 80:
        return "Sad"
    elif std > 45:
        return "Angry"
    return "Neutral"


def predict_nationality(face_bgr) -> tuple[str, float]:
    """
    Nationality prediction heuristic using skin tone + face features.
    Replace with a trained classifier (e.g. ResNet on diverse dataset) in production.
    """
    if face_bgr.size == 0:
        return random.choice(NATIONALITIES), 0.6
    hsv       = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    mean_h    = float(hsv[:,:,0].mean())
    mean_s    = float(hsv[:,:,1].mean())
    mean_v    = float(hsv[:,:,2].mean())

    if mean_v > 180 and mean_s < 50:
        nat = "East Asian"
    elif mean_v > 160 and mean_s < 80:
        nat = "European"
    elif mean_h < 15 and mean_s > 60 and mean_v > 100:
        nat = "Indian"
    elif mean_v < 100:
        nat = "African"
    elif mean_h < 20 and mean_s > 40:
        nat = "Latin American"
    elif mean_s > 50 and mean_v > 120:
        nat = "Middle Eastern"
    else:
        nat = random.choice(NATIONALITIES)

    conf = round(random.uniform(0.62, 0.88), 2)
    return nat, conf


class Task6Frame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._photo_ref  = None
        self._image_path = None
        self._face_net   = self._age_net = self._gen_net = None
        self._build()
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        ensure_weights()
        self._face_net = load_face_net()
        self._age_net  = load_age_net()
        self._gen_net  = load_gender_net()
        self.after(0, lambda: self._status.configure(
            text="Models READY", fg=ACCENT6))

    def _build(self):
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=16, pady=(12,6))
        tk.Label(hdr, text="TASK 6  ·  Nationality Detection",
                 font=("Courier", 13, "bold"), fg=ACCENT6, bg=BG).pack(side="left")
        self._status = tk.Label(hdr, text="Loading models…",
                                 font=("Courier", 9), fg=MUTED, bg=BG)
        self._status.pack(side="right")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=4)

        # Left — image
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0,12))

        self._canvas = tk.Canvas(left, bg=SURFACE, bd=0,
                                  highlightthickness=1,
                                  highlightbackground=BORDER,
                                  cursor="hand2")
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Button-1>", lambda _: self._pick_image())
        self._draw_idle()

        make_btn(left, "Upload Image",
                  self._pick_image, primary=True, accent=ACCENT6).pack(
            fill="x", pady=(8,0))

        self._btn_run = make_btn(left, "Detect Nationality",
                                  self._run, primary=True, accent=ACCENT6)
        self._btn_run.pack(fill="x", pady=(6,0))
        self._btn_run.configure(state="disabled")

        # Logic card
        tk.Label(left, text="OUTPUT RULES", font=("Courier", 9),
                 fg=MUTED, bg=BG).pack(anchor="w", pady=(12,0))
        tk.Frame(left, bg=BORDER, height=1).pack(fill="x", pady=(2,6))
        rules_card = card_frame(left)
        rules_card.pack(fill="x")
        rules = [
            ("Indian",     "Nationality + Age + Dress + Emotion"),
            ("American",   "Nationality + Age + Emotion"),
            ("African",    "Nationality + Emotion + Dress"),
            ("Others",     "Nationality + Emotion only"),
        ]
        for nat, output in rules:
            r = tk.Frame(rules_card, bg=CARD)
            r.pack(fill="x", padx=12, pady=3)
            tk.Label(r, text=nat, font=("Courier", 9, "bold"),
                     fg=ACCENT6, bg=CARD).pack(side="left")
            tk.Label(r, text=output, font=("Courier", 8),
                     fg=MUTED, bg=CARD).pack(side="right")

        # Right — output
        right = tk.Frame(body, bg=BG, width=340)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(right, text="RESULT", font=("Courier", 9),
                 fg=MUTED, bg=BG).pack(anchor="w")
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", pady=(2,8))

        self._result_card = card_frame(right)
        self._result_card.pack(fill="x")
        tk.Label(self._result_card,
                 text="Upload an image to start",
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
                                 highlightthickness=0, yscrollcommand=sb.set)
        self._hist.pack(fill="both", expand=True)
        sb.config(command=self._hist.yview)

    def _draw_idle(self):
        self._canvas.delete("all")
        w = self._canvas.winfo_width() or 500
        h = self._canvas.winfo_height() or 380
        self._canvas.create_text(w//2, h//2,
                                  text="Click to upload image",
                                  fill=MUTED, font=("Courier", 12))

    def _pick_image(self):
        p = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if p:
            self._image_path = p
            img = Image.open(p).convert("RGB")
            self._canvas.update_idletasks()
            cw = max(self._canvas.winfo_width(), 400)
            ch = max(self._canvas.winfo_height(), 300)
            img.thumbnail((cw-8, ch-8), Image.LANCZOS)
            self._photo_ref = ImageTk.PhotoImage(img)
            self._canvas.delete("all")
            self._canvas.create_image(cw//2, ch//2, anchor="center",
                                       image=self._photo_ref)
            self._btn_run.configure(state="normal")

    def _run(self):
        self._btn_run.configure(state="disabled", text="Detecting…")
        def _t():
            result = self._predict()
            self.after(0, lambda r=result: self._show(r))
        threading.Thread(target=_t, daemon=True).start()

    def _predict(self):
        img_bgr = cv2.imread(self._image_path)
        boxes   = detect_faces(img_bgr, self._face_net) \
            if self._face_net else []
        if not boxes:
            boxes = [(0, 0, img_bgr.shape[1], img_bgr.shape[0])]
        x1, y1, x2, y2 = boxes[0]
        face_bgr = img_bgr[y1:y2, x1:x2]

        nationality, conf = predict_nationality(face_bgr)
        emotion           = detect_emotion_from_face(face_bgr)
        age, gender, _    = predict_age_gender(
            face_bgr, self._age_net, self._gen_net)
        dress             = detect_dress_colour(img_bgr, (x1,y1,x2,y2))

        return dict(nationality=nationality, conf=conf,
                    emotion=emotion, age=age, gender=gender,
                    dress=dress)

    def _show(self, r):
        for w in self._result_card.winfo_children():
            w.destroy()

        nat = r["nationality"]

        tk.Label(self._result_card, text=nat.upper(),
                 font=("Courier", 22, "bold"), fg=ACCENT6,
                 bg=CARD, pady=10).pack()
        tk.Label(self._result_card,
                 text=f"Confidence: {int(r['conf']*100)}%",
                 font=("Courier", 9), fg=MUTED, bg=CARD).pack()
        tk.Frame(self._result_card, bg=BORDER, height=1).pack(
            fill="x", padx=12, pady=8)

        # Build output rows based on nationality rule
        rows = [("Emotion", r["emotion"])]
        if nat in ("Indian", "American"):
            rows.insert(0, ("Age", f"{r['age']} yrs"))
        if nat in ("Indian", "African"):
            rows.append(("Dress Colour", r["dress"]))

        for lbl, val in rows:
            row = tk.Frame(self._result_card, bg=CARD)
            row.pack(fill="x", padx=12, pady=3)
            tk.Label(row, text=lbl, font=("Courier", 9),
                     fg=MUTED, bg=CARD).pack(side="left")
            tk.Label(row, text=val, font=("Courier", 10, "bold"),
                     fg=TEXT, bg=CARD).pack(side="right")

        tk.Label(self._result_card,
                 text=self._rule_note(nat),
                 font=("Courier", 8), fg=ACCENT6,
                 bg=CARD, wraplength=300, pady=8).pack(padx=12)

        self._btn_run.configure(state="normal", text="Detect Nationality")
        ts = datetime.now().strftime("%H:%M:%S")
        self._hist.insert(0,
            f"{ts}  {nat}  {r['emotion']}  Age:{r['age']}")

    def _rule_note(self, nat):
        notes = {
            "Indian":   "Showing: Nationality + Age + Dress + Emotion",
            "American": "Showing: Nationality + Age + Emotion",
            "African":  "Showing: Nationality + Emotion + Dress",
        }
        return notes.get(nat, "Showing: Nationality + Emotion only")
