import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"

import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

# Create all folders
Path(r"C:\STM32_OTA1\model").mkdir(parents=True,        exist_ok=True)
Path(r"C:\STM32_OTA1\corrections").mkdir(parents=True,   exist_ok=True)
Path(r"C:\STM32_OTA1\model\generated").mkdir(parents=True, exist_ok=True)
Path(r"C:\STM32_OTA1\invalid_samples").mkdir(parents=True, exist_ok=True)

NUM_CLASSES  = 11
MODEL        = r"C:\STM32_OTA1\model\mnist.keras"
BASE_MODEL   = r"C:\STM32_OTA1\model\mnist_base.keras"
IMG_FOLDER   = r"C:\Users\HP\Downloads\archive (5)\Img"
MAX_PER_CLASS = 300   # max images per letter (img011–img062)
                      # 52 classes × 300 = 15600 letter samples total

print("=" * 45)
print("  STM32 OTA1 — First Time Setup")
print("=" * 45)

# ── [1/4] Load MNIST ────────────────────────────────────
print("\n[1/4] Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32")[..., None] / 255.0
x_test  = x_test.astype("float32")[..., None]  / 255.0

print(f"      Train : {x_train.shape}")
print(f"      Test  : {x_test.shape}")

# ── [2/4] Load EMNIST letter PNGs (class 10 = Invalid) ──
print("\n[2/4] Loading EMNIST letter images (class 10 = invalid)...")
print(f"      Source: {IMG_FOLDER}")

all_files = sorted(os.listdir(IMG_FOLDER))

# Group files by prefix number (img011 → 11, img062 → 62)
from collections import defaultdict
class_files = defaultdict(list)
for fname in all_files:
    if not fname.endswith('.png'):
        continue
    try:
        prefix = int(fname[3:6])   
        if prefix >= 11:           
            class_files[prefix].append(fname)
    except ValueError:
        continue

print(f"      Letter classes found: {len(class_files)}  (expected 52)")

x_az_list = []
total_loaded = 0

for prefix in sorted(class_files.keys()):
    files = class_files[prefix][:MAX_PER_CLASS]
    for fname in files:
        img_path = os.path.join(IMG_FOLDER, fname)
        try:
            img = Image.open(img_path).convert('L')
            img = img.resize((28, 28), Image.LANCZOS)
            arr = np.array(img).astype("float32") / 255.0

            # Per-image inversion: MNIST = white digit on black bg
            if arr.mean() > 0.5:
                arr = 1.0 - arr

            x_az_list.append(arr)
            total_loaded += 1
        except Exception as e:
            print(f"      Skipped {fname}: {e}")

x_az = np.array(x_az_list)[..., np.newaxis]  # (N, 28, 28, 1)
y_az = np.full(len(x_az), 10, dtype=np.int32)
print(f"      Letter images loaded: {len(x_az)}")

# Split 10% of letters for validation
split      = int(0.9 * len(x_az))
x_az_train, x_az_val = x_az[:split], x_az[split:]
y_az_train, y_az_val = y_az[:split], y_az[split:]

# Add letter validation to MNIST test set
x_val = np.concatenate([x_test, x_az_val], axis=0)
y_val = np.concatenate([y_test, y_az_val], axis=0)
print(f"      Validation: {len(x_test)} digits + {len(x_az_val)} letters = {len(x_val)} total")

# ── [3/4] Combine & shuffle ─────────────────────────────
print("\n[3/4] Combining and shuffling dataset...")
x_all = np.concatenate([x_train, x_az_train], axis=0)
y_all = np.concatenate([y_train, y_az_train], axis=0).astype(np.int32)

idx = np.random.permutation(len(x_all))
x_all, y_all = x_all[idx], y_all[idx]
print(f"      Total training samples : {len(x_all)}")
print(f"      Total validation samples: {len(x_val)}")

# ── [4/4] Build & Train ─────────────────────────────────
print("\n[4/4] Building model...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
], name="mnist_model")

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

print("\n      Training... (5-6 min on i3, please wait)")
print("      Do NOT close this window\n")



cw = {i : 1.0 for i in range (10)}
cw[10] = 15.0
model.fit(
    x_all, y_all,
    epochs          = 5,
    batch_size      = 128,
    validation_data = (x_val, y_val),
    class_weight    = cw,
    verbose         = 1,
)

# ── Evaluate ────────────────────────────────────────────
loss, acc = model.evaluate(x_val, y_val, verbose=0)
print(f"\n  Overall Accuracy : {acc * 100:.2f}%")
print(f"  Loss             : {loss:.4f}")

# Per-class accuracy
print("\n  Per-class accuracy:")
preds = np.argmax(model.predict(x_val, verbose=0), axis=1)
for cls in range(11):
    idx_cls = np.where(y_val == cls)[0]
    if len(idx_cls) == 0:
        continue
    cls_acc = np.mean(preds[idx_cls] == cls) * 100
    label   = f"Digit {cls}" if cls < 10 else "Invalid(A-Z)"
    print(f"    {label:>12}: {cls_acc:.1f}%  ({len(idx_cls)} samples)")

# ── Save ────────────────────────────────────────────────
model.save(MODEL)
model.save(BASE_MODEL)

print("\n" + "=" * 45)
print("  ✅ SETUP COMPLETE")
print("=" * 45)
print(f"  mnist.keras       ← used by pipeline")
print(f"  mnist_base.keras  ← backup (never touched)")
print()
print("  Now run in order:")
print("  1.  python train.py          ← test fine-tune")
print("  2.  python auto_pipeline.py  ← start OTA")
print("=" * 45)