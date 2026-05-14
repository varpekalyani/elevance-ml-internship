"""
download_weights.py
===================
Downloads all required model weights from working mirrors.
Run: python download_weights.py
"""

import urllib.request
import os

WEIGHTS = "weights"
os.makedirs(WEIGHTS, exist_ok=True)

# Working URLs for all model files
files = [
    # Face detector (already have these but including for completeness)
    (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "deploy.prototxt"
    ),
    (
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "res10_300x300_ssd_iter_140000.caffemodel"
    ),
    # Age model — working mirror
    (
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel",
        "age_net.caffemodel"
    ),
    (
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_deploy.prototxt",
        "age_net.prototxt"
    ),
    # Gender model — working mirror
    (
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel",
        "gender_net.caffemodel"
    ),
    (
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_deploy.prototxt",
        "gender_net.prototxt"
    ),
]

print("=" * 50)
print("Downloading model weights...")
print("=" * 50)

for url, fname in files:
    dest = os.path.join(WEIGHTS, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"[OK]   Already exists: {fname}")
        continue
    print(f"[DL]   Downloading {fname} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"[DONE] {fname}  ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"[FAIL] {fname} — {e}")

print("\n" + "=" * 50)
print("Verifying weights folder:")
for f in os.listdir(WEIGHTS):
    size = os.path.getsize(os.path.join(WEIGHTS, f))
    print(f"  {f:50s}  {size/1024:.0f} KB")
print("=" * 50)