#!/usr/bin/env python3
"""Download the exact official MicroDuck policies used in validation."""
import hashlib
import os
from pathlib import Path
import urllib.request

MODELS = {
    "velstand.onnx": "ce894d30b3343036d499bfac1102bdaa58d59a21c1b58401ecb70fb6a2f63e0a",
    "alpha_stand.onnx": "1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b",
}

BASE = "https://huggingface.co/pollen-robotics/microduck-policies/resolve/v7/"
DEST = Path.home() / ".cache/duck-sim/policies/current"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name, expected in MODELS.items():
        path = DEST / name
        if path.exists():
            if not path.is_file():
                raise SystemExit(f"[STOP] Not a regular file: {path}")
            if sha256(path) != expected:
                raise SystemExit(
                    f"[STOP] Existing model differs; refusing overwrite: {path}"
                )
            print(f"[PASS] Existing verified model: {name}")
            continue

        temporary = DEST / (name + ".download")
        if temporary.exists():
            raise SystemExit(f"[STOP] Temporary file already exists: {temporary}")

        print(f"[DOWNLOAD] {name}")
        try:
            request = urllib.request.Request(
                BASE + name,
                headers={"User-Agent": "MicroDuck-Model-Setup"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                with temporary.open("xb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)

            if sha256(temporary) != expected:
                raise ValueError(f"SHA-256 mismatch: {name}")

            os.replace(temporary, path)
            print(f"[PASS] {name}")
        finally:
            temporary.unlink(missing_ok=True)

    print()
    print("Official walking and standing models: READY")
    print("PPO V3 model_99.onnx: NOT INCLUDED")
    print("See MODEL_SETUP.md for the required PPO V3 model.")
    print("Source: Pollen Robotics (Apache-2.0)")

if __name__ == "__main__":
    main()
