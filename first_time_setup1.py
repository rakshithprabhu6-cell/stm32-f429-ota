import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

# ── Folders ──────────────────────────────
Path(r"C:\STM32_OTA1\model").mkdir(parents=True, exist_ok=True)
Path(r"C:\STM32_OTA1\corrections").mkdir(parents=True, exist_ok=True)
Path(r"C:\STM32_OTA1\model\generated").mkdir(parents=True, exist_ok=True)
Path(r"C:\STM32_OTA1\invalid_samples").mkdir(parents=True, exist_ok=True)

MODEL       = r"C:\STM32_OTA1\model\mnist.keras"
BASE_MODEL  = r"C:\STM32_OTA1\model\mnist_base.keras"
INVALID_DIR = r"C:\STM32_OTA1\invalid_samples"
NUM_CLASSES = 11

print("=" * 50)
print("STM32 OTA1 — First Time Setup (11 classes)")
print("=" * 50)

# ── 1. MNIST digits (class 0-9) ───────────
def load_mnist():
    print("\n[1/2] Loading MNIST digits...")
    (X_train, y_train), (X_test, y_test) = \
        tf.keras.datasets.mnist.load_data()
    print(f"      Train : {X_train.shape}")
    print(f"      Test  : {X_test.shape}")
    return X_train, y_train, X_test, y_test

# ── 2. A-Z CSV (class 10) ─────────────────
def load_az_csv(count=None):
    print("\n[2/2] Loading A-Z Handwritten CSV...")
    path = r"C:\Users\HP\Downloads\archive (3)\A_Z Handwritten Data.csv"

    try:
        df = pd.read_csv(path, header=None)

        if count is not None:
            df = df.iloc[:count]

        # Columns 1-784 = pixels
        images = df.iloc[:, 1:].values.astype(np.uint8)
        images = images.reshape(-1, 28, 28)

        # Invert if white background
        if images.mean() > 127:
            images = 255 - images

        # All labeled as class 10 (invalid)
        y = np.full(len(images), 10, dtype=np.uint8)

        print(f"      A-Z loaded: {len(images)} samples")
        return images, y

    except FileNotFoundError:
        print("      File not found — skipping")
        print(r"      Expected: C:\Users\HP\Downloads\archive (3)\A_Z Handwritten Data.csv")
        return np.zeros((0,28,28), dtype=np.uint8), \
               np.array([], dtype=np.uint8)
    except Exception as e:
        print(f"      FAILED: {e}")
        return np.zeros((0,28,28), dtype=np.uint8), \
               np.array([], dtype=np.uint8)

# ── 3. Local board invalid samples ────────
def load_local_invalid():
    print("\n[+] Loading local board invalid samples...")
    X, y = [], []
    path = Path(INVALID_DIR)

    for f in path.glob("*.npy"):
        try:
            pixels = np.load(str(f))
            if pixels.shape == (28, 28):
                X.append(pixels.astype(np.uint8))
                y.append(10)
        except:
            pass

    print(f"      Local board: {len(X)} samples")
    if len(X) == 0:
        return np.zeros((0,28,28), dtype=np.uint8), \
               np.array([], dtype=np.uint8)
    return np.array(X, dtype=np.uint8), \
           np.array(y, dtype=np.uint8)

# ── Build model ───────────────────────────
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(32, 3,
            activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3,
            activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3,
            activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES,
            activation='softmax'),
    ], name="mnist_11class")
    return model

# ── Main setup ────────────────────────────
def setup():
    # Load all data
    X_mnist, y_mnist, X_test, y_test = load_mnist()
    X_az,    y_az    = load_az_csv(15000)
    X_local, y_local = load_local_invalid()

    # Combine invalid sources
    inv_X = [X_az]
    inv_y = [y_az]

    if len(X_local) > 0:
        inv_X.append(X_local)
        inv_y.append(y_local)

    # Filter empty
    inv_X = [x for x in inv_X if len(x) > 0]
    inv_y = [y for y in inv_y if len(y) > 0]

    X_invalid = np.concatenate(inv_X, axis=0)
    y_invalid = np.concatenate(inv_y, axis=0)

    # Combine digits + invalid
    X_all = np.concatenate([X_mnist, X_invalid], axis=0)
    y_all = np.concatenate([y_mnist, y_invalid], axis=0)

    # Summary
    print(f"\n{'='*50}")
    print(f"  Dataset Summary:")
    print(f"  MNIST digits : {len(X_mnist)}")
    print(f"  A-Z CSV      : {len(X_az)}")
    print(f"  Local board  : {len(X_local)}")
    print(f"  Invalid total: {len(X_invalid)}")
    print(f"  Grand total  : {len(X_all)}")
    print(f"{'='*50}\n")

    # Normalize
    X_all  = X_all.reshape(-1,28,28,1).astype(np.float32)/255.0
    X_test = X_test.reshape(-1,28,28,1).astype(np.float32)/255.0

    # One-hot encode 11 classes
    y_all_cat  = tf.keras.utils.to_categorical(y_all,  NUM_CLASSES)
    y_test_cat = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    # Shuffle
    idx       = np.random.permutation(len(X_all))
    X_all     = X_all[idx]
    y_all_cat = y_all_cat[idx]

    # Build and compile
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]

    # Train
    print("\nTraining started... please wait (~10 min)")
    model.fit(
        X_all, y_all_cat,
        epochs=15,
        batch_size=128,
        validation_data=(X_test, y_test_cat),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n  Test accuracy : {acc*100:.2f}%")
    print(f"  Test loss     : {loss:.4f}")

    # Save
    model.save(MODEL)
    model.save(BASE_MODEL)

    print("\n" + "="*50)
    print("  SETUP COMPLETE!")
    print("="*50)
    print("  mnist.keras      ← used by pipeline")
    print("  mnist_base.keras ← backup")
    print()
    print("  Now run:")
    print("  python auto_pipeline1.py")
    print("="*50)

if __name__ == "__main__":
    setup()