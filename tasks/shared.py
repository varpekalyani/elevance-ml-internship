"""
tasks/shared.py
===============
Shared constants, model loader, and helpers used by all 6 task frames.
"""

import os, urllib.request, cv2, numpy as np
from datetime import datetime

# ── Colours ───────────────────────────────────────────────────────────────────
BG      = "#0D0D0D"
SURFACE = "#1A1A1A"
CARD    = "#222222"
ACCENT  = "#C8FF00"
TEXT    = "#F0F0F0"
MUTED   = "#888888"
BORDER  = "#333333"
SENIOR  = "#FF6B35"
MALE_C  = "#35B5FF"
FEMALE_C= "#FF6EC7"

# ── Weights dir ───────────────────────────────────────────────────────────────
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")

DOWNLOADS = [
    ("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
     "deploy.prototxt", "Face proto"),
    ("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
     "res10_300x300_ssd_iter_140000.caffemodel", "Face model"),
    ("https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_net.prototxt",
     "age_net.prototxt", "Age proto"),
    ("https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_net.caffemodel",
     "age_net.caffemodel", "Age model"),
    ("https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net.prototxt",
     "gender_net.prototxt", "Gender proto"),
    ("https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net.caffemodel",
     "gender_net.caffemodel", "Gender model"),
]

AGE_MIDS    = [1, 5, 10, 17, 28, 40, 50, 70]
GENDER_LIST = ["Male", "Female"]


def ensure_weights(callback=None):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    for url, fname, label in DOWNLOADS:
        dest = os.path.join(WEIGHTS_DIR, fname)
        if os.path.exists(dest):
            continue
        if callback:
            callback(f"Downloading {label}…")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            if callback:
                callback(f"[WARN] {label}: {e}")


def load_face_net():
    p = os.path.join(WEIGHTS_DIR, "deploy.prototxt")
    m = os.path.join(WEIGHTS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
    if os.path.exists(p) and os.path.exists(m):
        return cv2.dnn.readNetFromCaffe(p, m)
    return None


def load_age_net():
    p = os.path.join(WEIGHTS_DIR, "age_net.prototxt")
    m = os.path.join(WEIGHTS_DIR, "age_net.caffemodel")
    if os.path.exists(p) and os.path.exists(m):
        return cv2.dnn.readNetFromCaffe(p, m)
    return None


def load_gender_net():
    p = os.path.join(WEIGHTS_DIR, "gender_net.prototxt")
    m = os.path.join(WEIGHTS_DIR, "gender_net.caffemodel")
    if os.path.exists(p) and os.path.exists(m):
        return cv2.dnn.readNetFromCaffe(p, m)
    return None


def detect_faces(frame, face_net, conf_thresh=0.55):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                  1.0, (300, 300), (104, 177, 123))
    face_net.setInput(blob)
    dets  = face_net.forward()
    boxes = []
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf > conf_thresh:
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def predict_age_gender(face_img, age_net, gen_net):
    blob = cv2.dnn.blobFromImage(cv2.resize(face_img, (227, 227)),
                                  1.0, (227, 227), (78.43, 87.77, 114.90),
                                  swapRB=False)
    age = 25
    if age_net:
        age_net.setInput(blob)
        age = AGE_MIDS[int(np.argmax(age_net.forward()[0]))]
    gender, conf = "Male", 0.5
    if gen_net:
        gen_net.setInput(blob)
        gp     = gen_net.forward()[0]
        idx    = int(np.argmax(gp))
        gender = GENDER_LIST[idx]
        conf   = float(gp[idx])
    return age, gender, round(conf, 2)


# ── Shared button factory ─────────────────────────────────────────────────────
import tkinter as tk

def make_btn(parent, text, cmd, primary=False, danger=False,
             accent=ACCENT):
    if danger:
        bg, fg, abg = "#3A0000", "#FF4444", "#5A0000"
    elif primary:
        bg, fg, abg = accent, BG, "#A8D900"
    else:
        bg, fg, abg = CARD, TEXT, BORDER
    return tk.Button(parent, text=text, command=cmd,
                     bg=bg, fg=fg, activebackground=abg,
                     activeforeground=fg,
                     font=("Courier", 9, "bold"),
                     relief="flat", bd=0, padx=10, pady=8,
                     cursor="hand2")


def make_label(parent, text, font_size=10, color=TEXT, bold=False):
    weight = "bold" if bold else "normal"
    return tk.Label(parent, text=text,
                    font=("Courier", font_size, weight),
                    fg=color, bg=BG)


def card_frame(parent, **kwargs):
    return tk.Frame(parent, bg=CARD,
                    highlightbackground=BORDER,
                    highlightthickness=1, **kwargs)
