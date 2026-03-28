import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from pathlib import Path

# ── Paths (relative — runs inside GitHub Actions) ────────
CORRECTIONS = Path("corrections")
MODEL_PATH  = Path("model/mnist.keras")
BASE_PATH   = Path("model/mnist_base.keras")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 11


def load_corrections():
    """Load all .npy files from corrections/ folder"""
    cx, cy = [], []

    for f in sorted(CORRECTIONS.glob("*.npy")):
        try:
            pixels = np.load(str(f)).astype("float32")

            # Reshape if needed
            if pixels.ndim == 2:
                pixels = pixels[..., np.newaxis]     # (28,28,1)
            elif pixels.shape == (784,):
                pixels = pixels.reshape(28, 28, 1)

            pixels = pixels / 255.0 if pixels.max() > 1.0 else pixels

            # ── Determine label from filename ────────────
            stem = f.stem   # e.g. label3_1234 or invalid_1234
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

    return np.array(cx), np.array(cy, dtype=np.int32)


def retrain():
    print("=" * 45)
    print("  Cloud Retrain — GitHub Actions")
    print("=" * 45)

    # ── [1] Load MNIST ───────────────────────────────────
    print("\n[1/4] Loading MNIST (1000 per digit)...")
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
    print(f"      MNIST samples : {len(sx)}")

    # ── [2] Load corrections (digits + invalid) ──────────
    print("\n[2/4] Loading corrections...")
    cx, cy = load_corrections()

    digit_count   = int(np.sum(cy < 10))  if len(cy) else 0
    invalid_count = int(np.sum(cy == 10)) if len(cy) else 0
    print(f"      Digit corrections  : {digit_count}")
    print(f"      Invalid corrections: {invalid_count}")

    if len(cx) > 0:
        # Repeat corrections to boost their weight
        rep = max(5, 50 // len(cx))
        cx  = np.repeat(cx, rep, axis=0)
        cy  = np.repeat(cy, rep, axis=0)
        sx  = np.concatenate([sx, cx])
        sy  = np.concatenate([sy, cy])
        print(f"      Corrections added  : {len(cx)} (×{rep} repeats)")
    else:
        print("      No corrections found")

    # ── [3] Add synthetic invalid (random noise) ─────────
    # Ensures model never forgets invalid class even without letters
    print("\n[3/4] Adding synthetic invalid samples...")
    np.random.seed(42)
    x_noise = np.random.rand(2000, 28, 28, 1).astype("float32")
    y_noise = np.full(2000, 10, dtype=np.int32)
    sx = np.concatenate([sx, x_noise])
    sy = np.concatenate([sy, y_noise])
    print(f"      Noise samples added: 2000")

    # ── Shuffle ──────────────────────────────────────────
    idx  = np.random.permutation(len(sx))
    sx   = sx[idx]
    sy   = sy[idx].astype(np.int32)
    print(f"\n      Total training samples: {len(sx)}")

    # ── [4] Load or build model ──────────────────────────
    print("\n[4/4] Loading model...")
    if MODEL_PATH.exists():
        print("      Fine-tuning existing model...")
        model = tf.keras.models.load_model(str(MODEL_PATH))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        epochs = 3
    elif BASE_PATH.exists():
        print("      Loading base model...")
        model = tf.keras.models.load_model(str(BASE_PATH))
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        epochs = 5
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
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ], name="mnist_model")
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        epochs = 5

    # ── Train ────────────────────────────────────────────
    print(f"\n      Training {epochs} epoch(s)...\n")
    model.fit(
        sx, sy,
        epochs          = epochs,
        batch_size      = 128,
        validation_data = (x_te, y_te),
        verbose         = 1,
    )

    # ── Evaluate ─────────────────────────────────────────
    loss, acc = model.evaluate(x_te, y_te, verbose=0)
    print(f"\n  Digit Accuracy : {acc * 100:.2f}%")
    print(f"  Loss           : {loss:.4f}")

    # ── Save ─────────────────────────────────────────────
    model.save(str(MODEL_PATH))
    print(f"\n  ✅ Saved → {MODEL_PATH}")


if __name__ == "__main__":
    retrain()
