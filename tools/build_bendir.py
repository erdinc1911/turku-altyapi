"""Bendir örnek paketlerini VSCO 2 Community Edition (CC0) kayıtlarından üretir.

Kullanım:  python tools/build_bendir.py <VSCO wav klasörü>

Bendir kaydı olmadığı için en yakın sesler katmanlanır (synth.PERC_KITS içinde):
  gövde  : büyük tek derili davul, yumuşak vuruş (BDrumNewhit)
  vızıltı: kirişli tenor davul (timpsnare_tenor) -> bendirin kiriş vızıltısı
  parmak : konga parmak vuruşu (Conga-Tap1)
Her pakette tekrarı kırmak için 2 çeşitleme (round-robin) bulunur.
"""

import json
import os
import subprocess
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "app", "src", "main", "python", "altyapi", "samples")
SR = 44100

PACKS = {
    "bendir_govde": (["BDrumNewhit_v3_rr1_Sum.wav", "BDrumNewhit_v3_rr2_Sum.wav"], 1.0),
    "bendir_vizilti": (["timpsnare_tenor1.wav", "timpsnare_tenor2.wav"], 0.8),
    "bendir_parmak": (["Conga-Tap1_v1_rr1_Sum.wav", "Conga-Tap1_v1_rr2_Sum.wav"], 0.4),
}


def load(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, "<f4").copy()
    # baştaki sessizliği at (vuruş anı örneğin başında olsun)
    thr = np.abs(x).max() * 0.02
    start = max(0, int(np.argmax(np.abs(x) > thr)) - int(0.002 * SR))
    return x[start:]


def main(src):
    os.makedirs(OUT, exist_ok=True)
    for name, (files, max_sec) in PACKS.items():
        arrays, meta = {}, []
        for i, fn in enumerate(files):
            x = load(os.path.join(src, fn))[: int(max_sec * SR)]
            fade = int(0.03 * SR)
            x[-fade:] *= np.linspace(1, 0, fade)
            x = x / np.abs(x).max() * 0.95
            arrays["a%d" % i] = (x * 32767).astype(np.int16)
            meta.append({"key_lo": 60, "key_hi": 60, "root": 60, "tune_cents": 0, "sr": SR, "loop": None,
                         "peak": 0.95})
        arrays["meta"] = np.array(json.dumps(meta))
        path = os.path.join(OUT, name + ".npz")
        np.savez_compressed(path, **arrays)
        print("%-16s %d çeşit  %5.0f KB" % (name, len(files), os.path.getsize(path) / 1024))


if __name__ == "__main__":
    main(sys.argv[1])
