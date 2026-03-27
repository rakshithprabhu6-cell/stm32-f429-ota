import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import serial
import numpy as np
import shutil, time, subprocess
from pathlib import Path
from threading import Thread, Event
from train1 import retrain_model


# CONFIG

UART_PORT = "COM4"
BAUD      = 115200

CUBIDE    = r"C:\ST\STM32CubeIDE_2.0.0\STM32CubeIDE\stm32cubeide.exe"
PROGRAMMER = r"C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe"

WORKSPACE = r"C:\Users\HP\.stm32cubeaistudio\workspace"
PROJECT   = "stm32_f429"
BIN_PATH  = fr"{WORKSPACE}\{PROJECT}\STM32CubeIDE\Debug\{PROJECT}.elf"
AI_APP_DIR = fr"{WORKSPACE}\{PROJECT}\AI\App"

CORRECTIONS = Path(r"C:\STM32_OTA1\corrections")
MODEL       = Path(r"C:\STM32_OTA1\model\mnist.keras")
GEN_DIR     = Path(r"C:\STM32_OTA1\model\generated")

MAGIC       = 0xAB
SAMPLE_SIZE = 2 + 28 * 28

CORRECTIONS.mkdir(parents=True, exist_ok=True)
GEN_DIR.mkdir(parents=True, exist_ok=True)

pipeline_busy = Event()

#  Retrain 
def step1_train():
    return retrain_model()

#  STEdgeAI convert 
STEDGEAI = r"C:\ST\STEdgeAI\3.0\Utilities\windows\stedgeai.exe"

def step2_stedgeai():
    print("[STEDGEAI] Converting model...")
    r = subprocess.run([
        STEDGEAI, "generate",
        "--model", str(MODEL),
        "--target", "stm32f4",
        "--output", str(GEN_DIR),
        "--name", "network",
    ], capture_output=True, text=True, timeout=400)

    if r.returncode == 0:
        print("[STEDGEAI] ✓ Done!")
        return True
    else:
        print("[STEDGEAI] ✗ Failed")
        print(r.stdout[-1000:])
        print(r.stderr[-500:])
        return False

#  Copy generated files 
def step3_copy():
    files = ["network.c", "network.h", "network_data.c", "network_data.h"]
    dst_dir = Path(AI_APP_DIR)
    copied = 0
    for fname in files:
        src = GEN_DIR / fname
        dst = dst_dir / fname
        if src.exists():
            shutil.copy(str(src), str(dst))
            print(f"[COPY] {fname} → {dst_dir}")
            copied += 1
        elif dst.exists():
            print(f"[COPY] {fname} already in project folder")
            copied += 1
    print(f"[COPY] {copied}/{len(files)} files ready")
    return copied > 0

#  Build firmware (automatic)
def step4_build():
    print("[BUILD] Starting headless build...")
    r = subprocess.run([
        CUBIDE,
        "--launcher.suppressErrors",
        "-nosplash",
        "-application", "org.eclipse.cdt.managedbuilder.core.headlessbuild",
        "-data", WORKSPACE,
        "-build", PROJECT,
    ], capture_output=True, text=True, timeout=400)

    # Check for .elf or .bin as success indicator
    elf = Path(BIN_PATH)
    bin_alt = Path(BIN_PATH.replace(".elf", ".bin"))

    if r.returncode == 0 or elf.exists() or bin_alt.exists():
        print("[BUILD] ✓ Build succeeded")
        return True
    else:
        print("[BUILD] ✗ Failed")
        print(r.stdout[-1500:])
        print(r.stderr[-500:])
        return False

#  Flash board (automatic) 
def step5_flash():
    # Find the binary
    elf = Path(BIN_PATH)
    bin_file = Path(BIN_PATH.replace(".elf", ".bin"))

    if bin_file.exists():
        fw = str(bin_file)
    elif elf.exists():
        fw = str(elf)
    else:
        print(f"[FLASH] ERROR: No firmware found at {BIN_PATH}")
        return False
    print(f"[FLASH] Flashing {fw} ...")
    r = subprocess.run([
        PROGRAMMER,
        "-c", "port=SWD",
        "-w", fw, "0x08000000",
        "-v",
        "-rst",
    ], capture_output=True, text=True, timeout=400)

    if r.returncode == 0:
        print("[FLASH] ✓ Board flashed and reset!")
        return True
    else:
        print("[FLASH] ✗ Flash failed")
        print(r.stdout[-1000:])
        print(r.stderr[-300:])
        return False

# Full pipeline
def run_pipeline():
    if pipeline_busy.is_set():
        print("[PIPELINE] Already running — sample saved for next run")
        return
    pipeline_busy.set()
    t0 = time.time()
    print("\n" + "=" * 45)
    print("  OTA PIPELINE STARTED")
    print("=" * 45)

    for name, fn in [
        ("1/5 Retrain model  ", step1_train),
        ("2/5 Convert model  ", step2_stedgeai),
        ("3/5 Copy files     ", step3_copy),
        ("4/5 Build firmware ", step4_build),
        ("5/5 Flash board    ", step5_flash),
    ]:
        print(f"\n→ {name}")
        t = time.time()
        result = fn()
        if result is False:
            print(f"\n✗ PIPELINE STOPPED at: {name}")
            pipeline_busy.clear()
            return
        print(f"  ✓ done in {time.time()-t:.0f}s")

    print(f"\n{'=' * 45}")
    print(f"  ✅ COMPLETE in {time.time()-t0:.0f}s")
    print(f"{'=' * 45}\n")
    pipeline_busy.clear()

#  UART listener 
def listen_uart():
    try:
        ser = serial.Serial(UART_PORT, BAUD, timeout=5)
    except serial.SerialException as e:
        print(f"[ERROR] Cannot open {UART_PORT}: {e}")
        return

    print("=" * 45)
    print("  STM32 OTA PIPELINE READY")
    print(f"  Port    : {UART_PORT} @ {BAUD} baud")
    print(f"  Project : {PROJECT}")
    print("=" * 45)
    print("  On board:")
    print("  1. Draw a digit")
    print("  2. Tap PREDICT")
    print("  3. Tap WRONG")
    print("  4. Tap correct digit")
    print("  Pipeline starts automatically")
    print("=" * 45 + "\n")

    buf = bytearray()
    while True:
        try:
            buf += ser.read(ser.in_waiting or 1)
            while len(buf) >= SAMPLE_SIZE:
                idx = buf.find(MAGIC)
                if idx == -1:
                    buf.clear()
                    break
                if idx > 0:
                    print(f"[UART] Skipping {idx} garbage bytes")
                    buf = buf[idx:]
                if len(buf) < SAMPLE_SIZE:
                    break
                label  = buf[1]
                pixels = np.frombuffer(buf[2:2+28*28], dtype=np.uint8).reshape(28,28).copy()
                buf    = buf[SAMPLE_SIZE:]
                if label > 10:
                    print(f"[UART] Invalid label {label}, skipping")
                    continue
                ts    = int(time.time() * 1000)
                if label==10:
                    inv_dir=(r"C\STM32_OTA1\invalid_samples")
                    inv_dir.mkdir(exist_ok=True)
                    fname=inv_dir/f"invalid_{ts}.npy"
                    np.save(str(fname),pixels)
                    print(f"[RECV] Invalid sample saved")
                else:    
                    fname = CORRECTIONS / f"label{label}_{ts}.npy"
                    np.save(str(fname), pixels)
                    print(f"[RECV] ✓ label={label} saved")
                
        except serial.SerialException as e:
            print(f"[UART] Error: {e} — retrying in 2s")
            time.sleep(2)

if __name__ == "__main__":
    if not MODEL.exists():
        print("=" * 45)
        print("  ERROR: No model found!")
        print("  Run: python first_time_setup.py")
        print("=" * 45)
    else:
        print(f"[OK] Model found: {MODEL}")
        listen_uart()