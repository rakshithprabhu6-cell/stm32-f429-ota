import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]   = "0"
os.environ["TF_NUM_INTEROP_THREADS"]  = "4"
os.environ["TF_NUM_INTRAOP_THREADS"]  = "4"

import sys, io
sys.stdout =io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr =io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import tensorflow as tf
from pathlib import Path

CORRECTIONS = Path(r"C:\STM32_OTA1\corrections")
MODEL       = Path(r"C:\STM32_OTA1\model\mnist.keras")
BASE_MODEL  = Path(r"C:\STM32_OTA1\model\mnist_base.keras")


def retrain_model():

    #  Safety check 
    if not MODEL.exists() and not BASE_MODEL.exists():
        print("[TRAIN] ERROR: No model found!")
        print("[TRAIN] Run first_time_setup1.py ")
        return False

    print("[TRAIN] Loading MNIST subset (10k)...")
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
    x_tr = x_tr.astype("float32")[..., None] / 255.0
    x_te = x_te.astype("float32")[..., None] / 255.0

    # Balanced 1000 per digit = 10k total 
    sx, sy = [], []
    for d in range(10):
        idx = np.where(y_tr == d)[0][:1000]
        sx.append(x_tr[idx])
        sy.append(y_tr[idx])
    sx = np.concatenate(sx)
    sy = np.concatenate(sy)
    print(f"[TRAIN] Base samples: {sx.shape[0]}")

    #  Load corrections
    cx, cy = [], []
    for f in CORRECTIONS.glob("*.npy"):
        try:
            px    = np.load(str(f)).astype("float32")[..., None] / 255.0
            label = int(f.stem.split("_")[0].replace("label", ""))
            if 0 <= label <= 9:
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
        print(f"[TRAIN] Corrections: {len(cy)} samples added "
              f"(repeated x{rep})")
    else:
        print("[TRAIN] No corrections yet")

    # Shuffle
    idx = np.random.permutation(len(sx))
    sx  = sx[idx]
    sy  = sy[idx]
    print(f"[TRAIN] Total training samples: {sx.shape[0]}")

    #  Load model 
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

    #  Train 
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

    #  Save 
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL))

    # Save base backup once only — never overwrite
    if not BASE_MODEL.exists():
        model.save(str(BASE_MODEL))

    loss, acc = model.evaluate(x_te, y_te, verbose=0)
    print(f"[TRAIN] Accuracy : {acc  * 100:.2f}%")
    print(f"[TRAIN] Loss     : {loss:.4f}")
    print(f"[TRAIN] Saved    : {MODEL}")
    return True

#  THIS BLOCK WAS MISSING — this is why it did nothing

if __name__ == "__main__":
    print("=" * 40)
    print("  train.py — manual test run")
    print("=" * 40)

    ok = retrain_model()

    if ok:
        print("\n✅ train1.py finished successfully")
        
    else:
        print("\n✗ train.py FAILED")
        print("  Run first_time_setup1.py first")