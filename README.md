# Elevance Skills — ML Internship Project
### All 6 Tasks in One Unified Application

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-DNN-green)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-orange)
![Deadline](https://img.shields.io/badge/Deadline-22%2F07%2F2026-red)

---

##📌Overview
### This project integrates six machine learning tasks into a unified tabbed GUI application. Built with Python, OpenCV, and Tkinter, it demonstrates applied ML skills across computer vision, audio analysis, and rule‑based logic

---

## 📂 Project Structure

```
Elevance skills/
├── main_app.py              ← Launch point — opens tabbed GUI
├── requirements.txt
├── tasks/
│   ├── __init__.py
│   ├── shared.py            ← Shared models, colours, helpers
│   ├── task1_hair_gender.py
│   ├── task2_senior.py
│   ├── task3_voice.py
│   ├── task4_signlang.py
│   ├── task5_car_colour.py
│   └── task6_nationality.py
├── weights/                 ← Auto-downloaded on first run
└── logs/                    ← CSV + Excel output (Task 2)

```



## ⚡Quick Start

```bash
pip install -r requirements.txt
python main_app.py
```

---

## 🧩Tasks

### Task 1 — Long Hair Gender Detection
- Upload image or webcam snap
- Age 20–30: Long hair → Female, Short hair → Male
- Outside 20–30: standard gender prediction
- **Models:** SSD Face + AgeNet + GenderNet (Caffe)

### Task 2 — Senior Citizen Identification
- Real-time webcam or video file
- Detects multiple persons simultaneously
- Age > 60 → SENIOR CITIZEN (orange bounding box)
- Logs age, gender, timestamp → CSV + styled Excel
- **Models:** SSD Face + AgeNet + GenderNet (Caffe)

### Task 3 — Age & Emotion Detection via Voice
- Upload `.wav`/`.mp3` or record 5 seconds
- Female voice → rejected ("Upload male voice.")
- Age ≤ 60 → age only
- Age > 60 → age + emotion + SENIOR label
- **Library:** librosa (F0 pitch for gender, MFCCs for age, ZCR for emotion)

### Task 4 — Sign Language Detection
- **Only active 6 PM – 10 PM** (time-locked)
- Upload image or live webcam
- Detects ASL hand signs A–Z + common words
- Skin segmentation + HOG features

### Task 5 — Car Colour Detection
- Upload image or video/webcam
- Blue cars → RED rectangle
- Other colour cars → BLUE rectangle
- Counts people at signal
- **Detector:** YOLO (if weights present) or HOG fallback

### Task 6 — Nationality Detection
- Upload image → nationality + emotion
- Indian: + age + dress colour
- American: + age
- African: + dress colour
- Others: nationality + emotion only

---

## 🧠Models Used

| Model | Purpose | Source |
|-------|---------|--------|
| SSD ResNet-10 | Face detection | OpenCV pretrained |
| AgeNet (Caffe) | Age estimation | Levi & Hassner 2015 |
| GenderNet (Caffe) | Gender classification | Levi & Hassner 2015 |
| HOGDescriptor | Person + hand detection | OpenCV built-in |
| librosa pyin | Voice F0 / pitch | librosa |

All weights download automatically on first run.

---

## 📬Contact

Email: training@elevanceskills.com  
Include: Name, Domain, GitHub Link
