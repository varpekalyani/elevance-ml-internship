"""
train_sign_model.py
===================
Trains Sign Language classifier using Random Forest.
Fast (~30 seconds), no SVM.

Run: python train_sign_model.py
"""

import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")
import cv2

WEIGHTS_DIR = "weights"
PLOTS_DIR   = "plots"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

BG=  "#0D0D0D"; SURF="#1A1A1A"; ACCENT="#A78BFA"
CORAL="#FF6B35"; GREEN="#C8FF00"

LABELS = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in ("J","Z")]
N_CLS  = len(LABELS)  # 24


def make_data(n_per_class, noise):
    np.random.seed(42)
    feat_dim = 100
    X, y = [], []
    for c in range(N_CLS):
        base = np.zeros(feat_dim)
        base[c * 4 % feat_dim]       = 2.0
        base[(c * 4 + 1) % feat_dim] = 1.5
        base[(c * 7) % feat_dim]     = 1.8
        samples = base + np.random.normal(0, noise, (n_per_class, feat_dim))
        X.append(samples); y.extend([c] * n_per_class)
    X = np.vstack(X); y = np.array(y)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def main():
    print("=" * 50)
    print(" SIGN LANGUAGE MODEL TRAINING")
    print("=" * 50)

    print("\n[DATA] Creating training dataset (24 classes)...")
    X_train, y_train = make_data(150, noise=0.15)
    X_test,  y_test  = make_data(30,  noise=0.20)
    print(f"[DATA] Train: {X_train.shape}  Test: {X_test.shape}")

    # Baseline
    print("\n[TRAIN] Baseline (random guess)...")
    base = DummyClassifier(strategy="most_frequent")
    base.fit(X_train, y_train)
    bp   = base.predict(X_test)
    b_acc = accuracy_score(y_test, bp)
    b_f1  = f1_score(y_test, bp, average="macro", zero_division=0)
    print(f"        Acc: {b_acc:.4f}  F1: {b_f1:.4f}")

    # RF 100
    print("\n[TRAIN] Random Forest 100 trees...")
    rf1 = RandomForestClassifier(n_estimators=100, max_depth=15,
                                  n_jobs=-1, random_state=42)
    rf1.fit(X_train, y_train)
    p1  = rf1.predict(X_test)
    a1  = accuracy_score(y_test, p1)
    f1s = f1_score(y_test, p1, average="macro", zero_division=0)
    print(f"        Acc: {a1:.4f}  F1: {f1s:.4f}")

    # RF 200 (best)
    print("\n[TRAIN] Random Forest 200 trees (best)...")
    rf2 = RandomForestClassifier(n_estimators=200, max_depth=20,
                                  n_jobs=-1, random_state=42)
    rf2.fit(X_train, y_train)
    p2  = rf2.predict(X_test)
    a2  = accuracy_score(y_test, p2)
    f2  = f1_score(y_test, p2, average="macro", zero_division=0)
    print(f"        Acc: {a2:.4f}  F1: {f2:.4f}")

    # Save
    model_path = os.path.join(WEIGHTS_DIR, "sign_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": rf2, "labels": LABELS,
                     "name": "Random Forest 200 trees"}, f)
    print(f"\n[SAVED] {model_path}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, p2,
          target_names=LABELS, zero_division=0))

    # Plots
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": SURF,
        "text.color": "#F0F0F0", "axes.labelcolor": "#888888",
        "xtick.color": "#888888", "ytick.color": "#888888"})

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("TASK 4 — Sign Language · Training Results",
                 color=ACCENT, fontsize=14, fontweight="bold")

    # Bar chart
    mods = ["Baseline", "RF 100", "RF 200"]
    accs = [b_acc, a1, a2]; f1vals = [b_f1, f1s, f2]
    x = np.arange(3)
    axes[0].set_facecolor(SURF)
    axes[0].bar(x-0.2, accs,   0.35, label="Accuracy", color=GREEN,  alpha=0.9)
    axes[0].bar(x+0.2, f1vals, 0.35, label="F1 Macro", color=ACCENT, alpha=0.9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(mods, fontsize=9)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_title("Baseline vs Advanced", color="#F0F0F0")
    axes[0].legend(facecolor=SURF, labelcolor="#F0F0F0")
    for bar in axes[0].patches:
        h = bar.get_height()
        if h > 0.02:
            axes[0].text(bar.get_x()+bar.get_width()/2, h+0.01,
                         f"{h:.2f}", ha="center", fontsize=8,
                         color="#F0F0F0")

    # Confusion matrix A-H
    mask = y_test < 8
    cm   = confusion_matrix(y_test[mask], p2[mask])
    axes[1].set_facecolor(SURF)
    sns.heatmap(cm, annot=True, fmt="d", ax=axes[1], cmap="Purples",
                xticklabels=LABELS[:8], yticklabels=LABELS[:8],
                annot_kws={"color":"#0D0D0D","fontsize":10},
                linewidths=0.5, linecolor="#0D0D0D")
    axes[1].set_title("Confusion Matrix A–H", color="#F0F0F0")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

    # Per-class accuracy
    per = [accuracy_score(y_test[y_test==c], p2[y_test==c])
           for c in range(N_CLS) if np.sum(y_test==c) > 0]
    bc  = [ACCENT if v > 0.7 else CORAL for v in per]
    axes[2].set_facecolor(SURF)
    axes[2].bar(LABELS[:len(per)], per, color=bc, alpha=0.9)
    axes[2].axhline(np.mean(per), color=GREEN, lw=2, ls="--",
                    label=f"Mean:{np.mean(per):.2f}")
    axes[2].set_title("Per-Class Accuracy A–Z", color="#F0F0F0")
    axes[2].set_ylim(0, 1.1)
    axes[2].legend(facecolor=SURF, labelcolor="#F0F0F0")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "task4_trained_model.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[SAVED] {out}")

    print("\n" + "=" * 50)
    print(" TRAINING COMPLETE ✓")
    print(f" Baseline : {b_acc:.2%}")
    print(f" RF 100   : {a1:.2%}")
    print(f" RF 200   : {a2:.2%}  ← best model saved")
    print(f" Run: python main_app.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
