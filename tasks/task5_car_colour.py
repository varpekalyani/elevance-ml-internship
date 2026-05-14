"""
tasks/task5_car_colour.py
Task 5: Car Colour Detection at Traffic Signal
- Detect cars and their colours
- Blue cars → RED rectangle
- Other colour cars → BLUE rectangle
- Count people at signal
- GUI with image preview
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2, numpy as np, threading, os
from datetime import datetime
from tasks.shared import (BG, SURFACE, CARD, TEXT, MUTED, BORDER,
                           make_btn, card_frame)

ACCENT5 = "#FBBF24"

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")

# ── Colour detection in HSV ────────────────────────────────────────────────────
COLOUR_RANGES = {
    "Blue":   [(np.array([100,100, 50]), np.array([130,255,255]))],
    "Red":    [(np.array([  0,100, 50]), np.array([ 10,255,255])),
               (np.array([170,100, 50]), np.array([180,255,255]))],
    "White":  [(np.array([  0,  0,200]), np.array([180, 30,255]))],
    "Black":  [(np.array([  0,  0,  0]), np.array([180,255, 50]))],
    "Silver": [(np.array([  0,  0,150]), np.array([180, 30,200]))],
    "Yellow": [(np.array([ 20,100, 50]), np.array([ 35,255,255]))],
    "Green":  [(np.array([ 36,100, 50]), np.array([ 86,255,255]))],
}


def detect_colour(roi_bgr) -> str:
    if roi_bgr.size == 0:
        return "Unknown"
    hsv   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    best, best_count = "Unknown", 0
    for colour, ranges in COLOUR_RANGES.items():
        count = 0
        for lo, hi in ranges:
            count += int(cv2.countNonZero(cv2.inRange(hsv, lo, hi)))
        if count > best_count:
            best, best_count = colour, count
    return best


def load_yolo_or_hog():
    """
    Try loading YOLO for car/person detection.
    Falls back to HOG pedestrian + basic car detection.
    """
    yolo_cfg   = os.path.join(WEIGHTS_DIR, "yolov3.cfg")
    yolo_wts   = os.path.join(WEIGHTS_DIR, "yolov3.weights")
    yolo_names = os.path.join(WEIGHTS_DIR, "coco.names")

    if all(os.path.exists(p) for p in [yolo_cfg, yolo_wts, yolo_names]):
        net   = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_wts)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        with open(yolo_names) as f:
            classes = [l.strip() for l in f.readlines()]
        return "yolo", net, classes

    # Fallback: HOG person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return "hog", hog, None


def detect_with_hog(frame, hog):
    """Detect people using HOG, detect cars by area heuristic."""
    h, w = frame.shape[:2]
    # HOG people
    rects, _ = hog.detectMultiScale(
        frame, winStride=(8,8), padding=(4,4), scale=1.05)
    people = [(x, y, x+bw, y+bh) for (x, y, bw, bh) in rects]

    # Simple car heuristic: large horizontal rectangles in lower half
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cars = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)
        area   = cw * ch
        if (area > 8000 and 1.2 < aspect < 4.5
                and y > h * 0.25):
            cars.append((x, y, x+cw, y+ch))
    return cars, people


def detect_with_yolo(frame, net, classes):
    h, w = frame.shape[:2]
    blob  = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416),
                                   swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    out_layers  = [layer_names[i-1]
                   for i in net.getUnconnectedOutLayers().flatten()]
    outputs = net.forward(out_layers)

    cars, people = [], []
    for out in outputs:
        for det in out:
            scores = det[5:]
            cls_id = int(np.argmax(scores))
            conf   = float(scores[cls_id])
            if conf < 0.4:
                continue
            label = classes[cls_id]
            cx, cy, bw, bh = (det[:4] * np.array([w,h,w,h])).astype(int)
            x1, y1 = cx - bw//2, cy - bh//2
            x2, y2 = cx + bw//2, cy + bh//2
            if label in ("car","truck","bus","motorbike"):
                cars.append((x1,y1,x2,y2))
            elif label == "person":
                people.append((x1,y1,x2,y2))
    return cars, people


class Task5Frame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._photo_ref = None
        self._detector  = None
        self._det_type  = None
        self._classes   = None
        self._running   = False
        self._cap       = None
        self._build()
        threading.Thread(target=self._load_detector, daemon=True).start()

    def _load_detector(self):
        self.after(0, lambda: self._status.configure(
            text="Loading detector…", fg=MUTED))
        self._det_type, self._detector, self._classes = load_yolo_or_hog()
        label = "YOLO READY" if self._det_type == "yolo" else "HOG READY (no YOLO)"
        self.after(0, lambda: self._status.configure(
            text=label, fg=ACCENT5))

    def _build(self):
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=16, pady=(12,6))
        tk.Label(hdr, text="TASK 5  ·  Car Colour Detection",
                 font=("Courier", 13, "bold"), fg=ACCENT5, bg=BG).pack(side="left")
        self._status = tk.Label(hdr, text="Loading…",
                                 font=("Courier", 9), fg=MUTED, bg=BG)
        self._status.pack(side="right")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=4)

        # Left — image/video canvas
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0,12))

        self._canvas = tk.Canvas(left, bg=SURFACE, bd=0,
                                  highlightthickness=1,
                                  highlightbackground=BORDER)
        self._canvas.pack(fill="both", expand=True)
        self._draw_idle()

        btn_row = tk.Frame(left, bg=BG)
        btn_row.pack(fill="x", pady=(8,0))
        make_btn(btn_row, "Upload Image",
                  self._pick_image, primary=True, accent=ACCENT5).pack(
            side="left", expand=True, fill="x", padx=(0,6))
        make_btn(btn_row, "Load Video", self._load_video).pack(
            side="left", expand=True, fill="x", padx=(0,6))
        self._btn_cam = make_btn(btn_row, "Webcam", self._start_webcam)
        self._btn_cam.pack(side="left", expand=True, fill="x", padx=(0,6))
        self._btn_stop = make_btn(btn_row, "Stop", self._stop, danger=True)
        self._btn_stop.pack(side="left", expand=True, fill="x")
        self._btn_stop.configure(state="disabled")

        # Right — stats
        right = tk.Frame(body, bg=BG, width=300)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(right, text="DETECTION STATS", font=("Courier", 9),
                 fg=MUTED, bg=BG).pack(anchor="w")
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", pady=(2,8))

        stats = card_frame(right)
        stats.pack(fill="x")
        self._stat_lbls = {}
        for key, label, color in [
            ("cars",    "Total cars",    ACCENT5),
            ("blue",    "Blue cars",     "#3B82F6"),
            ("other",   "Other colour",  TEXT),
            ("people",  "People at signal", "#C8FF00"),
        ]:
            row = tk.Frame(stats, bg=CARD)
            row.pack(fill="x", padx=12, pady=4)
            tk.Label(row, text=label, font=("Courier", 9),
                     fg=MUTED, bg=CARD).pack(side="left")
            lbl = tk.Label(row, text="0",
                           font=("Courier", 15, "bold"),
                           fg=color, bg=CARD)
            lbl.pack(side="right")
            self._stat_lbls[key] = lbl

        tk.Label(right, text="LEGEND", font=("Courier", 9),
                 fg=MUTED, bg=BG).pack(anchor="w", pady=(16,0))
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", pady=(2,8))
        leg = card_frame(right)
        leg.pack(fill="x")
        for sym, color, desc in [
            ("■", "#FF0000", "Red box = Blue car"),
            ("■", "#0000FF", "Blue box = Other cars"),
            ("■", "#00C864", "Green box = Person"),
        ]:
            r = tk.Frame(leg, bg=CARD)
            r.pack(fill="x", padx=12, pady=4)
            tk.Label(r, text=sym, fg=color, bg=CARD,
                     font=("Courier", 14)).pack(side="left")
            tk.Label(r, text=f"  {desc}", font=("Courier", 8),
                     fg=TEXT, bg=CARD).pack(side="left")

        tk.Label(right, text="COLOUR LOG", font=("Courier", 9),
                 fg=MUTED, bg=BG).pack(anchor="w", pady=(16,0))
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", pady=(2,6))
        lf = tk.Frame(right, bg=BG)
        lf.pack(fill="both", expand=True)
        sb = tk.Scrollbar(lf, bg=BG, troughcolor=SURFACE)
        sb.pack(side="right", fill="y")
        self._log = tk.Listbox(lf, bg=SURFACE, fg=TEXT,
                                font=("Courier", 8), bd=0,
                                highlightthickness=0, yscrollcommand=sb.set)
        self._log.pack(fill="both", expand=True)
        sb.config(command=self._log.yview)

    def _draw_idle(self):
        self._canvas.delete("all")
        w = self._canvas.winfo_width() or 500
        h = self._canvas.winfo_height() or 380
        self._canvas.create_text(w//2, h//2,
                                  text="Upload image or load video",
                                  fill=MUTED, font=("Courier", 12))

    def _pick_image(self):
        p = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not p:
            return
        frame = cv2.imread(p)
        result = self._process_frame(frame)
        self._show_frame(result)

    def _load_video(self):
        p = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")])
        if p:
            self._start_capture(p)

    def _start_webcam(self):
        self._start_capture(0)

    def _start_capture(self, source):
        if self._running:
            self._stop()
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            messagebox.showerror("Error", f"Cannot open: {source}")
            return
        self._running = True
        self._btn_cam.configure(state="disabled")
        self._btn_stop.configure(state="normal")
        threading.Thread(target=self._loop, daemon=True).start()

    def _stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self._btn_cam.configure(state="normal")
        self._btn_stop.configure(state="disabled")
        self.after(200, self._draw_idle)

    def _loop(self):
        skip = 0
        while self._running:
            if not self._cap:
                break
            ret, frame = self._cap.read()
            if not ret:
                self.after(0, self._stop)
                break
            skip += 1
            if skip % 5 == 0:
                frame = self._process_frame(frame)
            self._show_frame(frame)

    def _process_frame(self, frame):
        if self._detector is None:
            return frame

        if self._det_type == "yolo":
            cars, people = detect_with_yolo(frame, self._detector, self._classes)
        else:
            cars, people = detect_with_hog(frame, self._detector)

        blue_count  = 0
        other_count = 0
        ts = datetime.now().strftime("%H:%M:%S")

        for (x1,y1,x2,y2) in cars:
            roi    = frame[max(0,y1):y2, max(0,x1):x2]
            colour = detect_colour(roi)
            if colour == "Blue":
                rect_color = (0, 0, 255)   # RED box for blue car
                blue_count += 1
            else:
                rect_color = (255, 0, 0)   # BLUE box for others
                other_count += 1
            cv2.rectangle(frame, (x1,y1), (x2,y2), rect_color, 2)
            cv2.putText(frame, colour, (x1, max(y1-6,12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, rect_color, 2)
            self.after(0, lambda c=colour, t=ts:
                       self._log.insert(0, f"{t}  Car: {c}"))

        for (x1,y1,x2,y2) in people:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,100), 2)
            cv2.putText(frame, "Person", (x1, max(y1-6,12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,100), 2)

        # Counts overlay
        cv2.putText(frame,
                    f"Cars:{len(cars)} Blue:{blue_count} People:{len(people)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,0), 2)

        self.after(0, lambda c=len(cars), b=blue_count,
                   o=other_count, p=len(people):
                   self._update_stats(c, b, o, p))
        return frame

    def _update_stats(self, cars, blue, other, people):
        self._stat_lbls["cars"].configure(text=str(cars))
        self._stat_lbls["blue"].configure(text=str(blue))
        self._stat_lbls["other"].configure(text=str(other))
        self._stat_lbls["people"].configure(text=str(people))

    def _show_frame(self, frame_bgr):
        self._canvas.update_idletasks()
        cw = max(self._canvas.winfo_width(), 400)
        ch = max(self._canvas.winfo_height(), 320)
        rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img   = Image.fromarray(rgb)
        img.thumbnail((cw, ch), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._photo_ref = photo
        self.after(0, lambda p=photo: self._put(p))

    def _put(self, photo):
        self._canvas.delete("all")
        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        self._canvas.create_image(w//2, h//2, anchor="center", image=photo)

    def on_close(self):
        self._stop()
