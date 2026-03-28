import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]   = "0"
os.environ["TF_NUM_INTEROP_THREADS"]  = "4"
os.environ["TF_NUM_INTRAOP_THREADS"]  = "4"

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from collections import defaultdict

CORRECTIONS    = Path(r"C:\STM32_OTA1\corrections")
INVALID_FOLDER = Path(r"C:\STM32_OTA1\invalid_samples")
MODEL          = Path(r"C:\STM32_OTA1\model\mnist.keras")
BASE_MODEL     = Path(r"C:\STM32_OTA1\model\mnist_base.keras")
IMG_FOLDER     = r"C:\Users\HP\Downloads\archive (5)\Img"
MAX_PER_CLASS  = 100   # letters per class during fine-tune (keeps it fast)


def load_letter_samples(img_folder, max_per_class=100):
    """Load EMNIST letter PNGs as class 10 (Invalid)"""
    all_files = sorted(os.listdir(img_folder))
    class_files = defaultdict(list)
    for fname in all_files:
        if not fname.endswith('.png'):
            continue
        try:
            prefix = int(fname[3:6])
            if prefix >= 11:  # letters only
                class_files[prefix].append(fname)
        except ValueError:
            continue

    x_list = []
    for prefix in sorted(class_files.keys()):
        for fname in class_files[prefix][:max_per_class]:
            try:
                img = Image.open(os.path.join(img_folder, fname)).convert('L')
                img = img.resize((28, 28), Image.LANCZOS)
                arr = np.array(img).astype("float32") / 255.0
                if arr.mean() > 0.5:  # invert if black-on-white
                    arr = 1.0 - arr
                x_list.append(arr)
            except:
                continue

    x = np.array(x_list)[..., np.newaxis]
    y = np.full(len(x), 10, dtype=np.int32)
    return x, y


def retrain_model():

    # Safety check
    if not MODEL.exists() and not BASE_MODEL.exists():
        print("[TRAIN] ERROR: No model found!")
        print("[TRAIN] Run first_time_setup.py first")
        return False

    # ── [1] Load MNIST subset ────────────────────────────
    print("[TRAIN] Loading MNIST subset (1000 per digit)...")
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
    x_tr = x_tr.astype("float32")[..., None] / 255.0
    x_te = x_te.astype("float32")[..., None] / 255.0

    sx, sy = [], []
    for d in range(10):
        idx = np.where(y_tr == d)[0][:1000]
        sx.append(x_tr[idx])
        sy.append(y_tr[idx])
    sx = np.concatenate(sx)
    sy = np.concatenate(sy)
    print(f"[TRAIN] MNIST samples  : {sx.shape[0]}")

    # ── [2] Load letter samples (class 10) ───────────────
    print(f"[TRAIN] Loading letter samples (Invalid class)...")
    try:
        x_az, y_az = load_letter_samples(IMG_FOLDER, MAX_PER_CLASS)
        print(f"[TRAIN] Letter samples : {len(x_az)}  ({MAX_PER_CLASS} per class × 52 classes)")

        # Split 10% for validation
        split = int(0.9 * len(x_az))
        x_az_train, x_az_val = x_az[:split], x_az[split:]
        y_az_train, y_az_val = y_az[:split], y_az[split:]

        sx = np.concatenate([sx, x_az_train])
        sy = np.concatenate([sy, y_az_train])

        # Add letters to validation
        x_te = np.concatenate([x_te, x_az_val])
        y_te = np.concatenate([y_te, y_az_val])
        print(f"[TRAIN] Validation     : digits + {len(x_az_val)} letters")

    except Exception as e:
        print(f"[TRAIN] Letter samples skipped: {e}")

    # ── [3] Load corrections ─────────────────────────────
    cx, cy = [], []
    for f in CORRECTIONS.glob("*.npy"):
        try:
            px    = np.load(str(f)).astype("float32")[..., None] / 255.0
            label = int(f.stem.split("_")[0].replace("label", ""))
            if 0 <= label <= 10:   # ✅ include label 10 (invalid corrections)
                cx.append(px)
                cy.append(label)
        except Exception as e:
            print(f"[TRAIN] Skipping {f.name}: {e}")

    if cx:
        cx  = np.array(cx)
        cy  = np.array(cy)
        rep = max(5, 50 // len(cx))
        cx  = np.repeat(cx, rep, axis=0)
        cy  = np.repeat(cy, rep, axis=0)
        sx  = np.concatenate([sx, cx])
        sy  = np.concatenate([sy, cy])
        print(f"[TRAIN] Corrections    : {len(cy)} samples added (×{rep})")
    else:
        print("[TRAIN] No corrections yet")

    # ── [4] Shuffle ──────────────────────────────────────
    idx = np.random.permutation(len(sx))
    sx  = sx[idx]
    sy  = sy[idx].astype(np.int32)
    print(f"[TRAIN] Total training : {sx.shape[0]}")

    # ── [5] Load & compile model ─────────────────────────
    if MODEL.exists():
        print("[TRAIN] Fine-tuning existing model...")
        model = tf.keras.models.load_model(str(MODEL))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        epochs = 2
    else:
        print("[TRAIN] Loading base model...")
        model = tf.keras.models.load_model(str(BASE_MODEL))
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        epochs = 3

    # ── [6] Train ────────────────────────────────────────
    print(f"[TRAIN] Training {epochs} epoch(s)...")
    model.fit(
        sx, sy,
        epochs          = epochs,
        batch_size      = 64,
        validation_data = (x_te, y_te),
        verbose         = 1,
        callbacks       = [
            tf.keras.callbacks.EarlyStopping(
                monitor   = "val_accuracy",
                patience  = 2,
                min_delta = 0.001,
                verbose   = 1,
            )
        ],
    )

    # ── [7] Save ─────────────────────────────────────────
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL))

    if not BASE_MODEL.exists():
        model.save(str(BASE_MODEL))

    # ── [8] Evaluate per class ───────────────────────────
    loss, acc = model.evaluate(x_te, y_te, verbose=0)
    print(f"\n[TRAIN] Overall Accuracy : {acc * 100:.2f}%")
    print(f"[TRAIN] Loss             : {loss:.4f}")

    preds = np.argmax(model.predict(x_te, verbose=0), axis=1)
    print("\n[TRAIN] Per-class accuracy:")
    for cls in range(11):
        idx_cls = np.where(y_te == cls)[0]
        if len(idx_cls) == 0:
            continue
        cls_acc = np.mean(preds[idx_cls] == cls) * 100
        label   = f"Digit {cls}" if cls < 10 else "Invalid(A-Z)"
        print(f"         {label:>12}: {cls_acc:.1f}%  ({len(idx_cls)} samples)")

    print(f"\n[TRAIN] Saved: {MODEL}")
    return True


if __name__ == "__main__":
    print("=" * 40)
    print("  train.py — manual test run")
    print("=" * 40)

    ok = retrain_model()

    if ok:
        print("\n✅ train.py finished successfully")
    else:
        print("\n✗ train.py FAILED")
        print("  Run first_time_setup.py first")