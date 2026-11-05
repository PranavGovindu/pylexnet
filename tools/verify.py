"""The generated kernel must equal the generic interpreter over the same weights, byte for byte, on the dev split."""
import subprocess
import sys
from pathlib import Path

import score

HERE = Path(__file__).resolve().parent
fs = score.files(sys.argv[1] if len(sys.argv) > 1 else "dev")
bad = tot = 0
for p in fs:
    try:
        data = p.read_text("utf-8").replace("\r\n", "\n").encode()
    except UnicodeDecodeError:
        continue
    out = subprocess.run([str(HERE.parent / "pylexnet"), "--classes"], input=data, capture_output=True).stdout
    ref = subprocess.run([str(HERE.parent / "pylexnet"), "--interp", "--classes"], input=data, capture_output=True).stdout
    tot += len(data)
    if out != ref:
        bad += 1
        if bad < 4:
            i = next((j for j in range(min(len(out), len(ref))) if out[j] != ref[j]), -1)
            print("MISMATCH", p, i, out[i] if i >= 0 else len(out), ref[i] if i >= 0 else len(ref), data[max(0, i - 30):i + 10])
print(f"{len(fs)} files, {tot / 1e6:.1f} MB, files differing from rules: {bad}")
