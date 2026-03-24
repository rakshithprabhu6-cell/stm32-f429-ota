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
import random
import cv2

#  Folders 
Path(r"C:\STM32_OTA1\model").mkdir(parents=True, exist_ok=True)
Path(r"C:\STM32_OTA1\corrections").mkdir(parents=True, exist_ok=True)
Path(r"C:\STM32_OTA1\model\generated").mkdir(parents=True, exist_ok=True)
Path(r"C:\STM32_OTA1\invalid_samples").mkdir(parents=True, exist_ok=True)

MODEL      = r"C:\STM32_OTA1\model\mnist.keras"
BASE_MODEL = r"C:\STM32_OTA1\model\mnist_base.keras"
INVALID_DIR = r"C:\STM32_OTA1\invalid_samples"
NUM_CLASSES = 11

print("=" * 50)
print("STM32 OTA1 — First Time Setup (11 classes)")
print("=" * 50)

#  MNIST digi
def load_mnist():
    print("\n[1/4] Loading MNIST digits...")
    (X_train, y_train), (X_test, y_test) = \
        tf.keras.datasets.mnist.load_data()
    print(f"Train : {X_train.shape}")
    print(f"Test  : {X_test.shape}")
    return X_train, y_train, X_test, y_test

#  A-Z CSV (class 10) 
def load_az_csv(count=8000):
    print("\n[2/4] Loading A-Z Handwritten CSV...")
    path =(r"C:\Users\HP\Downloads\archive (3)\A_Z Handwritten Data.csv")

    try:
        df = pd.read_csv(path, header=None)
        images = df.iloc[:count, 1:].values
        images = images.reshape(-1, 28, 28).astype(np.uint8)
        y = np.full(len(images), 10, dtype=np.uint8)
        print(f"      A-Z loaded: {len(images)} samples")
        return images, y

    except FileNotFoundError:
        print("File not found — skipping")
        print(r"Expected: C:\Users\HP\Downloads\archive (3)\A_Z Handwritten Data.csv")
        return np.zeros((0,28,28), dtype=np.uint8), \
               np.array([], dtype=np.uint8)
    except Exception as e:
        print(f"FAILED: {e}")
        return np.zeros((0,28,28), dtype=np.uint8), \
               np.array([], dtype=np.uint8)

# ── 3. Math Symbols CSV (class 10) ───
def load_math_images(count=5000):
    print("\n[3/4] Loading Math Symbols (image folders)...")
    path = Path(r"C:\Users\HP\Downloads\archive (4)\dataset")
    X, y = [], []

    if not path.exists():
        print("Folder not found — skipping")
        print(r"Expected: C:\\Users\Hp\Downloads\archive(4)\datasets")
        return np.zeros((0,28,28), dtype=np.uint8), \
               np.array([], dtype=np.uint8)

    # Skip digit folders — only load symbols
    skip_folders = {'0','1','2','3','4',
                    '5','6','7','8','9'}

    for folder in path.iterdir():
        if not folder.is_dir(): continue
        if folder.name in skip_folders:
            print(f"Skipping digit folder: {folder.name}")
            continue

        folder_count = 0
        for img_file in folder.iterdir():
            if len(X) >= count: break
            if img_file.suffix.lower() not in \
               ['.png','.jpg','.jpeg']: continue

            img = cv2.imread(str(img_file),
                             cv2.IMREAD_GRAYSCALE)
            if img is None: continue

            img = cv2.resize(img, (28, 28))

            # Invert if white background
            if img.mean() > 127:
                img = 255 - img

            X.append(img)
            y.append(10)
            folder_count += 1

        if folder_count > 0:
            print(f"      {folder.name}: {folder_count} samples")
        if len(X) >= count: break

    print(f"Math total: {len(X)} samples")
    if len(X) == 0:
        return np.zeros((0,28,28), dtype=np.uint8), \
               np.array([], dtype=np.uint8)
    return np.array(X, dtype=np.uint8), \
           np.array(y, dtype=np.uint8)

# ── 4. Generated dots/lines (class 10) ────
def generate_invalid(count=2000):
    print("\n[4/4] Generating dots/lines/scribbles...")
    X_gen, y_gen = [], []

    for _ in range(count):
        canvas = np.zeros((28, 28), dtype=np.uint8)
        choice = random.randint(0, 3)

        if choice == 0:
            # Single dot
            cx = random.randint(5, 22)
            cy = random.randint(5, 22)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = cx+dx, cy+dy
                    if 0<=nx<28 and 0<=ny<28:
                        canvas[ny][nx] = 255

        elif choice == 1:
            # Horizontal line
            yp = random.randint(8, 20)
            x1 = random.randint(2, 10)
            x2 = random.randint(18, 26)
            for x in range(x1, x2):
                canvas[yp][x] = 255
                if yp+1 < 28:
                    canvas[yp+1][x] = 180

        elif choice == 2:
            # Diagonal line
            x1,py1 = random.randint(2,8),  random.randint(2,8)
            x2,py2 = random.randint(18,26), random.randint(18,26)
            for t in range(30):
                px = int(x1+(x2-x1)*t/30)
                py = int(py1+(py2-py1)*t/30)
                if 0<=px<28 and 0<=py<28:
                    canvas[py][px] = 255

        else:
            # Random scribble
            px = random.randint(5, 22)
            py = random.randint(5, 22)
            for _ in range(random.randint(10, 30)):
                px += random.randint(-3, 3)
                py += random.randint(-3, 3)
                px = max(0, min(27, px))
                py = max(0, min(27, py))
                canvas[py][px] = 255
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        nx, ny = px+dx, py+dy
                        if 0<=nx<28 and 0<=ny<28:
                            canvas[ny][nx] = 180

        X_gen.append(canvas)
        y_gen.append(10)

    print(f"Generated: {len(X_gen)} samples")
    return np.array(X_gen, dtype=np.uint8), \
           np.array(y_gen, dtype=np.uint8)

# ── 5. Local board invalid samples ────────
def load_local_invalid():
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
    print(f"Local board: {len(X)} samples")
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
    X_az,    y_az    = load_az_csv(8000)
    X_math, y_math = load_math_images(5000)
    X_dots,  y_dots  = generate_invalid(2000)
    X_local, y_local = load_local_invalid()

    # Combine invalid
    inv_X = [X_az, X_math, X_dots]
    inv_y = [y_az, y_math, y_dots]

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
    print(f"Dataset Summary:")
    print(f"MNIST digits : {len(X_mnist)}")
    print(f"A-Z CSV      : {len(X_az)}")
    print(f"Math CSV     : {len(X_math)}")
    print(f"Generated    : {len(X_dots)}")
    print(f"Local board  : {len(X_local)}")
    print(f"Invalid total: {len(X_invalid)}")
    print(f"Grand total  : {len(X_all)}")
    print(f"{'='*50}\n")

    # Normalize
    X_all  = X_all.reshape(-1,28,28,1).astype(np.float32)/255.0
    X_test = X_test.reshape(-1,28,28,1).astype(np.float32)/255.0

    # One-hot 11 classes
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
    print(f"\nTest accuracy : {acc*100:.2f}%")
    print(f"Test loss     : {loss:.4f}")

    # Save
    model.save(MODEL)
    model.save(BASE_MODEL)

    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    print(f"mnist.keras← used by pipeline")
    print(f"mnist_base.keras ← backup")
    print()
    print("Now run:")
    print("python auto_pipeline1.py")
    print("="*50)

if __name__ == "__main__":
    setup()


