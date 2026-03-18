import os, serial, numpy as np
import time, requests, base64, subprocess
from pathlib import Path
from threading import Thread


# UART,Git 
from dotenv import load_dotenv
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
UART_PORT    = "COM4"
RETRAIN_EVERY = 2
BAUD         = 115200

GITHUB_REPO  = "rakshithprabhu6-cell/stm32-mnist-ota"
CUBIDE = (
    r"C:\Users\HP\AppData\Local\Temp"
    r"\STM32CubeIDE_aeda5489-2276-46b0-8bc6-63f09bd9c873"
    r"\C\ST\STM32CubeIDE_1.19.0\STM32CubeIDE"
    r"\stm32cubeide.exe"
)
WORKSPACE = r"C:\Users\HP\.stm32cubeaistudio\workspace"
PROJECT   = "stm32_f429"
BIN_PATH  = fr"{WORKSPACE}\{PROJECT}\Debug\{PROJECT}.bin"


MAGIC       = 0xAB
SAMPLE_SIZE = 2 + 28 * 28
HEADERS     = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

def upload_sample(label, pixels):
    """Upload correction sample to GitHub repo"""
    ts      = int(time.time() * 1000)
    fname   = f"label{label}_{ts}.npy"
    content = base64.b64encode(pixels.tobytes()).decode()

    r = requests.put(
        f"https://api.github.com/repos/{GITHUB_REPO}"
        f"/contents/corrections/{fname}",
        headers=HEADERS,
        json={
            "message": f"Add correction: {fname}",
            "content": content
        }
    )
    ok = r.status_code in [200, 201]
    print(f"[UPLOAD] {'✓ OK' if ok else 'FAILED'} {fname}")
    return ok

def wait_for_release(sha_before):
    print("[CLOUD] GitHub Actions started...")
    print("[CLOUD] Waiting for new firmware (~2 min)...")

    last_seen_tag = None

    # Get current latest release tag before waiting
    try:
        r = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest",
            headers=HEADERS, timeout=15
        )
        if r.status_code == 200:
            last_seen_tag = r.json().get("tag_name")
            print(f"[CLOUD] Current release: {last_seen_tag or 'none'}")
    except:
        pass

    for attempt in range(72):   # max 12 minutes
        time.sleep(10)
        elapsed = (attempt + 1) * 10

        try:
            r = requests.get(
                f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest",
                headers=HEADERS, timeout=15
            )
            if r.status_code == 200:
                release = r.json()
                tag = release.get("tag_name")

                # New release detected
                if tag and tag != last_seen_tag:
                    for asset in release.get("assets", []):
                        if asset["name"].endswith(".keras"):
                            print(f"\n[CLOUD] ✓ New release: {tag}")
                            return asset["browser_download_url"]
                    # No asset yet - wait a bit more
                    print(f"[CLOUD] Release {tag} found but no asset yet...")

        except Exception as e:
            print(f"[CLOUD] Check error: {e}")

        mins = elapsed // 60
        secs = elapsed % 60
        print(f"[CLOUD] Still waiting... {mins}m {secs:02d}s")

    print("[CLOUD] Timeout")
    return None

def download_firmware(url):
    save_path = r"C:\STM32_OTA1\cloud_firmware.keras"
    print(f"[DOWNLOAD] Downloading model...")

    # Get asset ID from release instead of direct URL
    r = requests.get(
        f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest",
        headers=HEADERS,
        timeout=15
    )
    if r.status_code != 200:
        print(f"[DOWNLOAD] Cannot get release: {r.status_code}")
        return None

    release = r.json()
    asset_id = None
    for asset in release.get("assets", []):
        if asset["name"].endswith(".keras"):
            asset_id = asset["id"]
            break

    if not asset_id:
        print("[DOWNLOAD] No .keras asset found")
        return None

    # Download using asset ID with proper headers
    r = requests.get(
        f"https://api.github.com/repos/{GITHUB_REPO}/releases/assets/{asset_id}",
        headers={**HEADERS, "Accept": "application/octet-stream"},
        allow_redirects=True,
        timeout=60
    )
    if r.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(r.content)
        size = len(r.content) // 1024
        print(f"[DOWNLOAD] ✓ {size} KB saved")
        return save_path

    print(f"[DOWNLOAD] FAILED: {r.status_code}")
    return None

def flash_board(bin_path):
    print("[FLASH] Running full pipeline (convert+build+flash)...")

    import sys
    sys.path.insert(0, r"C:\STM32_OTA1")
    from auto_pipeline1 import step2_stedgeai, step3_copy, step4_build, step5_flash

    steps = [
        ("Convert", step2_stedgeai),
        ("Copy",    step3_copy),
        ("Build",   step4_build),
        ("Flash",   step5_flash),
    ]

    for name, fn in steps:
        print(f"[FLASH] → {name}...")
        if not fn():
            print(f"[FLASH] ✗ {name} failed")
            return False
        print(f"[FLASH] ✓ {name} done")

    return True
def get_current_sha():
    """Get latest commit SHA"""
    r = requests.get(
        f"https://api.github.com/repos/{GITHUB_REPO}"
        f"/commits/main",
        headers=HEADERS
    )
    if r.status_code == 200:
        return r.json()["sha"]
    return None

def handle_sample(label, pixels):
    print(f"\n[RECV] label={label}")
    print("[RECV] Uploading to GitHub...")

    sha = get_current_sha()
    if not upload_sample(label, pixels):
        return

    print("[RECV] ✓ Uploaded! GitHub Actions starting...")
    firmware_url = wait_for_release(sha)

    if firmware_url:
        bin_path = download_firmware(firmware_url)
        if bin_path:
            flash_board(bin_path)
            print("\n✅ Board updated with cloud-trained model!")
    else:
        print("[ERROR] No firmware received from cloud")
def listen():
    ser = serial.Serial(UART_PORT, BAUD, timeout=5)
    print("=" * 45)
    print("  STM32 CLOUD OTA READY")
    print(f"  Repo: {GITHUB_REPO}")
    print(f"  Retrain every: {RETRAIN_EVERY} errors")
    print("=" * 45)
    print("  Draw → PREDICT → WRONG → select digit")
    print("  Cloud retrains in ~4 min automatically")
    print("=" * 45 + "\n")

    buf = bytearray()
    corr_dir = Path(r"C:\STM32_OTA1\corrections")
    correction_count = 0

    while True:
        try:
            buf += ser.read(ser.in_waiting or 1)
            while len(buf) >= SAMPLE_SIZE:
                idx = buf.find(MAGIC)
                if idx == -1: buf.clear(); break
                if idx  > 0:  buf = buf[idx:]
                if len(buf) < SAMPLE_SIZE: break

                label  = buf[1]
                pixels = np.frombuffer(
                    buf[2:2+28*28], dtype=np.uint8
                ).reshape(28,28).copy()
                buf = buf[SAMPLE_SIZE:]

                if label > 9: continue

                correction_count += 1
                print(f"[RECV] ✓ label={label}")
                print(f"[RECV] Corrections: {correction_count}/{RETRAIN_EVERY}")

                # Save sample ONCE here (removed duplicate below)
                ts = int(time.time() * 1000)
                fname = f"label{label}_{ts}.npy"
                corr_dir.mkdir(parents=True, exist_ok=True)
                np.save(str(corr_dir / fname), pixels)
                print(f"[RECV] Saved locally: {fname}")

                if correction_count % RETRAIN_EVERY == 0:
                    print(f"[PIPELINE] {RETRAIN_EVERY} errors reached — retraining!")
                    correction_count = 0
                    Thread(
                        target=handle_sample,
                        args=(label, pixels),
                        daemon=True
                    ).start()
                else:
                    remaining = RETRAIN_EVERY - (correction_count % RETRAIN_EVERY)
                    print(f"[RECV] Need {remaining} more errors before retraining")

        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(1)

if __name__ == "__main__":
    listen()