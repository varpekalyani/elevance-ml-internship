"""
tasks/task1_hair_gender.py
Task 1: Long Hair → Female (age 20-30), Short Hair → Male (age 20-30)
Outside 20-30: standard gender prediction
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2, numpy as np, threading, time, os
from datetime import datetime
from tasks.shared import (BG, SURFACE, CARD, ACCENT, TEXT, MUTED, BORDER,
                           MALE_C, FEMALE_C, ensure_weights,
                           load_face_net, load_age_net, load_gender_net,
                           detect_faces, predict_age_gender,
                           make_btn, card_frame)

ACCENT1 = "#C8FF00"


def estimate_hair_length(img_rgb, face_box):
    x, y, w, h = face_box
    ih, iw = img_rgb.shape[:2]
    hair_region = img_rgb[max(0, y-h):y, max(0,x-w//4):min(iw,x+w+w//4)]
    body_region = img_rgb[min(ih,y+h):min(ih,y+h+h), max(0,x-w//4):min(iw,x+w+w//4)]

    def non_skin_ratio(region):
        if region.size == 0:
            return 0.0
        hsv  = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([0,20,70]), np.array([20,255,255]))
        return float(np.sum(cv2.bitwise_not(mask) > 0) / mask.size)

    hr = non_skin_ratio(hair_region)
    br = non_skin_ratio(body_region)
    return "Long" if (hr > 0.55 or hr - br > 0.15) else "Short"


class Task1Frame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._photo_ref = None
        self._image_path = None
        self._face_net = self._age_net = self._gen_net = None
        self._build()
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        ensure_weights()
        self._face_net = load_face_net()
        self._age_net  = load_age_net()
        self._gen_net  = load_gender_net()
        self.after(0, lambda: self._status.configure(
            text="Models READY", fg=ACCENT1))

    def _build(self):
        # Header row
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=16, pady=(12, 6))
        tk.Label(hdr, text="TASK 1  ·  Long Hair Gender Detection",
                 font=("Courier", 13, "bold"), fg=ACCENT1, bg=BG).pack(side="left")
        self._status = tk.Label(hdr, text="Loading models…",
                                font=("Courier", 9), fg=MUTED, bg=BG)
        self._status.pack(side="right")

        # Body
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=4)

        # Left — image preview
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))

        self._canvas = tk.Canvas(left, bg=SURFACE, bd=0,
                                  highlightthickness=1,
                                  highlightbackground=BORDER)
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Button-1>", lambda _: self._pick_image())
        self._draw_placeholder()

        btn_row = tk.Frame(left, bg=BG)
        btn_row.pack(fill="x", pady=(8, 0))
        make_btn(btn_row, "Upload Image", self._pick_image,
                 primary=True, accent=ACCENT1).pack(
            side="left", expand=True, fill="x", padx=(0, 6))
        make_btn(btn_row, "Webcam Snap", self._webcam_snap).pack(
            side="left", expand=True, fill="x")

        self._btn_run = make_btn(left, "Run Prediction",
                                  self._run_prediction, primary=True,
                                  accent=ACCENT1)
        self._btn_run.pack(fill="x", pady=(8, 0))
        self._btn_run.configure(state="disabled")

        # Right — results + logic
        right = tk.Frame(body, bg=BG, width=320)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(right, text="RESULT", font=("Courier", 9),
                 fg=MUTED, bg=BG).pack(anchor="w")
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", pady=(2, 8))

        self._result_card = card_frame(right)
        self._result_card.pack(fill="x")
        tk.Label(self._result_card, text="Upload an image to start",
                 font=("Courier", 10), fg=MUTED, bg=CARD, pady=16).pack()

        tk.Label(right, text="LOGIC", font=("Courier", 9),
                 fg=MUTED, bg=BG).pack(anchor="w", pady=(16, 0))
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", pady=(2, 8))

        logic = card_frame(right)
        logic.pack(fill="x")
        rules = [("Ages 20–30:", "Hair decides gender"),
                 ("  Long hair →", "FEMALE"),
                 ("  Short hair →", "MALE"),
                 ("Outside 20–30:", "Normal prediction")]
        for lbl, val in rules:
            r = tk.Frame(logic, bg=CARD)
            r.pack(fill="x", padx=12, pady=3)
            tk.Label(r, text=lbl, font=("Courier", 9),
                     fg=MUTED, bg=CARD).pack(side="left")
            tk.Label(r, text=val, font=("Courier", 9, "bold"),
                     fg=TEXT, bg=CARD).pack(side="right")

        tk.Label(right, text="HISTORY", font=("Courier", 9),
                 fg=MUTED, bg=BG).pack(anchor="w", pady=(16, 0))
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", pady=(2, 6))

        hist_f = tk.Frame(right, bg=BG)
        hist_f.pack(fill="both", expand=True)
        sb = tk.Scrollbar(hist_f, bg=BG, troughcolor=SURFACE)
        sb.pack(side="right", fill="y")
        self._hist = tk.Listbox(hist_f, bg=SURFACE, fg=TEXT,
                                 font=("Courier", 8), bd=0,
                                 highlightthickness=0,
                                 yscrollcommand=sb.set)
        self._hist.pack(fill="both", expand=True)
        sb.config(command=self._hist.yview)

    def _draw_placeholder(self):
        self._canvas.delete("all")
        self._canvas.update_idletasks()
        w = self._canvas.winfo_width() or 500
        h = self._canvas.winfo_height() or 380
        self._canvas.create_text(w//2, h//2,
                                  text="Click to upload image",
                                  fill=MUTED, font=("Courier", 12))

    def _pick_image(self):
        p = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if p:
            self._load_image(p)

    def _load_image(self, path):
        self._image_path = path
        img = Image.open(path).convert("RGB")
        self._canvas.update_idletasks()
        cw = max(self._canvas.winfo_width(), 400)
        ch = max(self._canvas.winfo_height(), 300)
        img.thumbnail((cw-8, ch-8), Image.LANCZOS)
        self._photo_ref = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        self._canvas.create_image(cw//2, ch//2, anchor="center",
                                   image=self._photo_ref)
        self._btn_run.configure(state="normal")

    def _webcam_snap(self):
        def _t():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.after(0, lambda: messagebox.showerror(
                    "Webcam", "Cannot access webcam"))
                return
            ret, frame = cap.read()
            cap.release()
            if ret:
                path = os.path.join(os.path.dirname(
                    os.path.dirname(__file__)), "webcam_snap.jpg")
                cv2.imwrite(path, frame)
                self.after(0, lambda: self._load_image(path))
        threading.Thread(target=_t, daemon=True).start()

    def _run_prediction(self):
        self._btn_run.configure(state="disabled", text="Analysing…")
        def _t():
            try:
                result = self._predict()
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.after(0, lambda: self._btn_run.configure(
                    state="normal", text="Run Prediction"))
                return
            self.after(0, lambda r=result: self._show_result(r))
        threading.Thread(target=_t, daemon=True).start()

    def _predict(self):
        img_bgr = cv2.imread(self._image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        boxes = detect_faces(img_bgr, self._face_net) if self._face_net else []
        if not boxes:
            boxes = [(0, 0, img_bgr.shape[1], img_bgr.shape[0])]
        x1, y1, x2, y2 = boxes[0]
        face_bgr = img_bgr[y1:y2, x1:x2]
        age, gender, conf = predict_age_gender(face_bgr, self._age_net, self._gen_net)
        hair = estimate_hair_length(img_rgb, (x1, y1, x2-x1, y2-y1))

        if 20 <= age <= 30:
            pred = "Female" if hair == "Long" else "Male"
            note = f"Age {age} in 20–30 → hair-based (inverted logic)"
        else:
            pred = gender
            note = f"Age {age} outside 20–30 → standard prediction"

        return dict(age=age, hair=hair, bio=gender,
                    pred=pred, conf=conf, note=note)

    def _show_result(self, r):
        for w in self._result_card.winfo_children():
            w.destroy()
        col = FEMALE_C if r["pred"] == "Female" else MALE_C
        tk.Label(self._result_card, text=r["pred"].upper(),
                 font=("Courier", 26, "bold"), fg=col, bg=CARD,
                 pady=10).pack()
        tk.Label(self._result_card,
                 text=f"Confidence: {int(r['conf']*100)}%",
                 font=("Courier", 9), fg=MUTED, bg=CARD).pack()
        tk.Frame(self._result_card, bg=BORDER, height=1).pack(
            fill="x", padx=12, pady=6)
        for lbl, val in [("Age detected", str(r["age"])),
                          ("Hair length",  r["hair"]),
                          ("Bio gender",   r["bio"])]:
            row = tk.Frame(self._result_card, bg=CARD)
            row.pack(fill="x", padx=12, pady=2)
            tk.Label(row, text=lbl, font=("Courier", 8),
                     fg=MUTED, bg=CARD).pack(side="left")
            tk.Label(row, text=val, font=("Courier", 9, "bold"),
                     fg=TEXT, bg=CARD).pack(side="right")
        tk.Label(self._result_card, text=r["note"],
                 font=("Courier", 8), fg=ACCENT1, bg=CARD,
                 wraplength=280, pady=8).pack(padx=12)

        ts = datetime.now().strftime("%H:%M:%S")
        self._hist.insert(0,
            f"{ts}  Age:{r['age']}  {r['hair']:5}  → {r['pred']}")
        self._btn_run.configure(state="normal", text="Run Prediction")
