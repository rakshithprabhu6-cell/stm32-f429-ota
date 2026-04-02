import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

# ── Paths (relative — runs inside GitHub Actions) ────────
CORRECTIONS = Path("corrections")
MODEL_PATH  = Path("model/mnist.keras")
BASE_PATH   = Path("model/mnist_base.keras")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 11

# How many MNIST samples per digit to use for fine-tune
MNIST_PER_DIGIT = 1000

# How many letter samples per A-Z class (200×26 = 5200 ≈ matches digit count)
LETTERS_PER_CLASS = 200


def load_corrections():
    """Load all .npy correction files from corrections/ folder."""
    cx, cy = [], []

    for f in sorted(CORRECTIONS.glob("*.npy")):
        try:
            pixels = np.load(str(f), allow_pickle=True).astype("float32")

            if pixels.ndim == 2:
                pixels = pixels[..., np.newaxis]
            elif pixels.shape == (784,):
                pixels = pixels.reshape(28, 28, 1)

            pixels = pixels / 255.0 if pixels.max() > 1.0 else pixels

            stem = f.stem
            if stem.startswith("invalid"):
                label = 10
            elif stem.startswith("label"):
                label = int(stem.split("_")[0].replace("label", ""))
            else:
                print(f"  Skipping unknown file: {f.name}")
                continue

            if 0 <= label <= 10:
                cx.append(pixels)
                cy.append(label)

        except Exception as e:
            print(f"  Skipping {f.name}: {e}")

    return (np.array(cx), np.array(cy, dtype=np.int32)) if cx else (np.array([]), np.array([]))


def load_invalid_samples():
    """Load A-Z handwritten letters as invalid class (class 10).
    Falls back to EMNIST, then random noise if CSV not available."""

    CSV_LOCAL = r"C:\Users\HP\Downloads\archive (8)\A_Z Handwritten Data\A_Z Handwritten Data.csv"
    CSV_CLOUD = "A_Z Handwritten Data.csv"
    CSV = CSV_LOCAL if os.path.exists(CSV_LOCAL) else \
          CSV_CLOUD  if os.path.exists(CSV_CLOUD) else None

    if CSV:
        print("      Loading A-Z CSV...")
        import pandas as pd
        df      = pd.read_csv(CSV, header=None)
        lbl_col = df.iloc[:, 0].values.astype(np.int32)
        pixels  = df.iloc[:, 1:].values.astype("float32") / 255.0
        x_az    = pixels.reshape(-1, 28, 28, 1)

        x_inv, y_inv = [], []
        for cls in range(26):
            idx = np.where(lbl_col == cls)[0][:LETTERS_PER_CLASS]
            x_inv.append(x_az[idx])
            y_inv.append(np.full(len(idx), 10, dtype=np.int32))

        x_inv = np.concatenate(x_inv)
        y_inv = np.concatenate(y_inv)
        print(f"      A-Z letters loaded : {len(x_inv)}  ({LETTERS_PER_CLASS}×26)")
        return x_inv, y_inv

    # EMNIST fallback
    print("      CSV not found — trying EMNIST fallback...")
    try:
        import tensorflow_datasets as tfds
        ds = tfds.load("emnist/letters", split="train", as_supervised=True)
        x_inv, y_inv = [], []
        for img, _ in ds.take(MNIST_PER_DIGIT * 10):
            x_inv.append(img.numpy().astype("float32") / 255.0)
            y_inv.append(10)
        x_inv = np.array(x_inv)
        y_inv = np.array(y_inv, dtype=np.int32)
        print(f"      EMNIST letters loaded: {len(x_inv)}")
        return x_inv, y_inv

    except Exception as e:
        print(f"      EMNIST failed ({e}) — using noise fallback")
        n     = MNIST_PER_DIGIT * 10
        x_inv = np.random.rand(n, 28, 28, 1).astype("float32")
        y_inv = np.full(n, 10, dtype=np.int32)
        print(f"      Noise fallback: {n} samples")
        return x_inv, y_inv


def retrain_model():
    print("=" * 50)
    print("  Cloud Retrain — GitHub Actions  (fixed)")
    print("=" * 50)

    # ── [1] Load MNIST ───────────────────────────────────
    print(f"\n[1/4] Loading MNIST ({MNIST_PER_DIGIT} per digit)...")
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
    x_tr = x_tr.astype("float32")[..., None] / 255.0
    x_te = x_te.astype("float32")[..., None] / 255.0

    sx, sy = [], []
    for d in range(10):
        idx = np.where(y_tr == d)[0][:MNIST_PER_DIGIT]
        sx.append(x_tr[idx])
        sy.append(y_tr[idx])
    sx = np.concatenate(sx)
    sy = np.concatenate(sy).astype(np.int32)
    print(f"      MNIST samples : {len(sx)}")

    # ── [2] Load corrections ─────────────────────────────
    print("\n[2/4] Loading corrections...")
    cx, cy = load_corrections()

    digit_count   = int(np.sum(cy < 10))  if len(cy) else 0
    invalid_count = int(np.sum(cy == 10)) if len(cy) else 0
    print(f"      Digit corrections  : {digit_count}")
    print(f"      Invalid corrections: {invalid_count}")

    if len(cx) > 0:
        # Repeat corrections to boost their influence
        rep = max(5, 50 // max(len(cx), 1))
        cx  = np.repeat(cx, rep, axis=0)
        cy  = np.repeat(cy, rep, axis=0)
        sx  = np.concatenate([sx, cx])
        sy  = np.concatenate([sy, cy])
        print(f"      Corrections added  : {len(cx)} (×{rep} repeats)")
    else:
        print("      No corrections found — training on MNIST + invalid only")

    # ── [3] Load invalid samples ─────────────────────────
    print("\n[3/4] Loading invalid samples (A-Z letters)...")
    x_inv, y_inv = load_invalid_samples()

    # Validation set: MNIST test + held-out invalid samples
    val_inv_count   = min(500, len(x_inv))
    x_val_combined  = np.concatenate([x_te,          x_inv[:val_inv_count]])
    y_val_combined  = np.concatenate([y_te,          y_inv[:val_inv_count]])

    # Training: add remaining invalid samples
    sx = np.concatenate([sx, x_inv[val_inv_count:]])
    sy = np.concatenate([sy, y_inv[val_inv_count:]])

    # ── Shuffle ──────────────────────────────────────────
    idx = np.random.permutation(len(sx))
    sx  = sx[idx]
    sy  = sy[idx].astype(np.int32)
    print(f"\n      Total training   : {len(sx)}")
    print(f"      Total validation : {len(x_val_combined)}")

    # Per-class counts
    print("\n      Samples per class:")
    for c in range(11):
        n     = int(np.sum(sy == c))
        label = f"Digit {c}" if c < 10 else "Invalid(A-Z)"
        print(f"        class {c:>2}  {label:>12} : {n}")

    # ── Balanced class weights (auto-computed) ────────────
    # This correctly handles whatever imbalance exists in sx/sy
    cw_arr = compute_class_weight(
        class_weight = 'balanced',
        classes      = np.arange(NUM_CLASSES),
        y            = sy
    )
    cw = dict(enumerate(cw_arr))
    print("\n      Class weights (auto-balanced):")
    for c, w in cw.items():
        label = f"Digit {c}" if c < 10 else "Invalid(A-Z)"
        print(f"        class {c:>2}  {label:>12} : {w:.3f}")

    # ── [4] Load or build model ──────────────────────────
    print("\n[4/4] Loading model...")
    if MODEL_PATH.exists():
        print("      Fine-tuning existing model...")
        model  = tf.keras.models.load_model(str(MODEL_PATH))
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss      = "sparse_categorical_crossentropy",
            metrics   = ["accuracy"],
        )
        epochs = 5

    elif BASE_PATH.exists():
        print("      Loading base model...")
        model  = tf.keras.models.load_model(str(BASE_PATH))
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss      = "sparse_categorical_crossentropy",
            metrics   = ["accuracy"],
        )
        epochs = 8

    else:
        print("      Building new model from scratch...")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ], name="mnist_model")
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss      = "sparse_categorical_crossentropy",
            metrics   = ["accuracy"],
        )
        epochs = 15

    # ── Train ────────────────────────────────────────────
    print(f"\n      Training {epochs} epoch(s)...\n")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor              = "val_accuracy",
            patience             = 3,
            restore_best_weights = True,
            verbose              = 1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 2,
            verbose  = 1,
        ),
    ]

    model.fit(
        sx, sy,
        epochs          = epochs,
        batch_size      = 128,
        validation_data = (x_val_combined, y_val_combined),
        class_weight    = cw,        # ← auto-balanced, NOT hardcoded 15.0
        callbacks       = callbacks,
        verbose         = 2,
    )

    # ── Evaluate ─────────────────────────────────────────
    print("\n  Per-class accuracy on validation set:")
    preds   = np.argmax(model.predict(x_val_combined, verbose=0), axis=1)
    all_ok  = True
    for cls in range(11):
        idx_cls = np.where(y_val_combined == cls)[0]
        if len(idx_cls) == 0:
            continue
        cls_acc = np.mean(preds[idx_cls] == cls) * 100
        label   = f"Digit {cls}" if cls < 10 else "Invalid(A-Z)"
        flag    = "" if cls_acc >= 90 else "  ⚠ LOW"
        if cls_acc < 90:
            all_ok = False
        print(f"    {label:>12} : {cls_acc:5.1f}%  ({len(idx_cls)} samples){flag}")

    loss, acc = model.evaluate(x_te, y_te, verbose=0)
    print(f"\n  MNIST Digit Accuracy : {acc * 100:.2f}%")
    print(f"  Loss                 : {loss:.4f}")

    # ── Save ─────────────────────────────────────────────
    model.save(str(MODEL_PATH))
    if all_ok:
        print(f"\n  ✅ Saved → {MODEL_PATH}  (all classes ≥ 90%)")
    else:
        print(f"\n  ⚠  Saved → {MODEL_PATH}  (some classes below 90%)")


if __name__ == "__main__":
    retrain_model()