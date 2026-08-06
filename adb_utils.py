
import subprocess
import time

import cv2
import numpy as np

ADB_PATH = "adb"
DEVICE_ID: str | None = None  # rempli par test_adb_connection()


def run_adb_command(cmd: list[str], binary: bool = False) -> subprocess.CompletedProcess:
    """Exécute une commande adb, ciblée sur DEVICE_ID si connu (utile si
    plusieurs devices/émulateurs sont connectés — avant, la commande partait
    toujours sans -s et pouvait viser le mauvais appareil)."""
    prefix = [ADB_PATH] + (["-s", DEVICE_ID] if DEVICE_ID else [])
    return subprocess.run(prefix + cmd, capture_output=True, text=not binary, shell=False)


def get_device_id() -> str | None:
    result = run_adb_command(["devices"])
    for line in result.stdout.strip().split("\n")[1:]:
        if "\tdevice" in line:
            return line.split("\t")[0]
    return None


def test_adb_connection() -> bool:
    global DEVICE_ID
    DEVICE_ID = get_device_id()
    if DEVICE_ID:
        print(f" ADB connecté : {DEVICE_ID}")
        return True
    print(" ADB non connecté")
    return False


def capture_phone_screen() -> np.ndarray | None:
    try:
        result = run_adb_command(["exec-out", "screencap", "-p"], binary=True)
        if result.returncode == 0 and result.stdout:
            img_array = np.frombuffer(result.stdout, dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f" Erreur capture: {e}")
    return None


def adb_tap(x: int, y: int, delay: float = 0.3) -> None:
    run_adb_command(["shell", "input", "tap", str(x), str(y)])
    time.sleep(delay)


def adb_long_press(x: int, y: int, duration_ms: int = 700, delay: float = 0.5) -> None:
    run_adb_command(["shell", "input", "swipe", str(x), str(y), str(x), str(y), str(duration_ms)])
    time.sleep(delay)


def adb_drag(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300, delay: float = 0.5) -> None:
    run_adb_command(["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)])
    time.sleep(delay)
