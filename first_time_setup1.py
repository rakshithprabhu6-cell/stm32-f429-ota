import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"

import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd

#  Create all folders 
Path(r"C:\STM32_OTA1\model").mkdir(parents=True,      exist_ok=True)
Path(r"C:\STM32_OTA1\corrections").mkdir(parents=True, exist_ok=True)
Path(r"C:\STM32_OTA1\model\generated").mkdir(parents=True, exist_ok=True)
Path(r"C:\STM32_OTA1\invalid_samples").mkdir(parents=True,exist_ok=True)
NUM_CLASSES=11

MODEL      = r"C:\STM32_OTA1\model\mnist.keras"
BASE_MODEL = r"C:\STM32_OTA1\model\mnist_base.keras"

print("=" * 45)
print("  STM32 OTA1 — First Time Setup")
print("=" * 45)

#  Load MNIST 
print("\n[1/3] Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = \
    tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32")[..., None] / 255.0
x_test  = x_test.astype("float32")[..., None]  / 255.0

print(f"      Train : {x_train.shape}")
print(f"      Test  : {x_test.shape}")

# Loading  A-Z dataset
print("\n[2/4]Loading A-Z samples (class 10=invalid)...") 
az_path=r"C:\Users\HP\Downloads\archive (6)\handwritten_data_785.csv"
try:
    df   = pd.read_csv(az_path, header=None).iloc[:15000]
    x_az = df.iloc[:, 1:].values.reshape(-1,28,28,1).astype("float32")/255.0
    y_az = np.full(len(x_az), 10, dtype=np.int32)
    print(f"      A-Z loaded: {len(x_az)} samples")
except Exception as e:
    print(f"      A-Z skipped: {e}")
    x_az = np.zeros((0,28,28,1), dtype="float32")
    y_az = np.array([], dtype=np.int32)

x_all = np.concatenate([x_train, x_az], axis=0)
y_all = np.concatenate([y_train, y_az], axis=0).astype(np.int32)
idx   = np.random.permutation(len(x_all))
x_all, y_all = x_all[idx], y_all[idx]

#   Build model 
print("\n[2/3] Building model...")
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(NUM_CLASSES,  activation="softmax"),
    ],
    name="mnist_model",
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

#  Train 
print("\n[3/3] Training... (5-6 min on i3, please wait)")
print("      Do NOT close this window\n")

model.fit(
    x_all, y_all,
    epochs          = 5,
    batch_size      = 128,
    validation_data = (x_test, y_test),
    verbose         = 1,
)

# Evaluate 
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n  Accuracy : {acc  * 100:.2f}%")
print(f"  Loss     : {loss:.4f}")

# Save both copies 
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