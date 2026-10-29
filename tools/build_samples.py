"""Ses bankasından uygulamanın kullandığı küçük örnek paketlerini üretir.

Kullanım:  python tools/build_samples.py FluidR3Mono_GM.sf3

Çıktı: app/src/main/python/altyapi/samples/<çalgı>.npz
Kaynak: FluidR3 Mono (MIT lisansı) — telif notu LICENSES/FluidR3Mono.txt içinde.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sf_reader import SoundFont  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "app", "src", "main", "python", "altyapi", "samples")

# çalgı adı -> (bank, program, en uzun süre sn, anahtar listesi [yalnız davul])
MELODIC = {
    "bas": (0, 33, 3.0),
    "yaylilar": (0, 48, 4.0),
    "ney": (0, 77, 4.0),
}
# davul kiti: ad -> GM tuşu
DRUMS = {
    "tumba": 64,      # alçak konga: bendir/darbuka düm gövdesi
    "slap": 62,       # kapalı konga vuruşu: bendir tek
    "timbale": 65,    # metalik çınlama: darbuka tek
    "bongo_rim": 60,  # kenar vuruşu: darbuka ka
    "zil": 54,        # tef zili: bendir zilleri için çok hafif katman
    "kudum": 41,      # alçak tom: derin bendir düm katmanı
}


def pick_zones(zones):
    """Her tuş aralığı için orta-yüksek hız katmanını seç, tekrarları at."""
    chosen = {}
    for z in zones:
        if not (z["vel_lo"] <= 100 <= z["vel_hi"]):
            continue
        key = (z["key_lo"], z["key_hi"])
        chosen.setdefault(key, z)
    return [chosen[k] for k in sorted(chosen)]


def to_int16(x):
    p = np.abs(x).max()
    if p > 0:
        x = x / p * 0.95
    return (x * 32767).astype(np.int16)


def save(name, zones_meta, audios):
    arrays = {"a%d" % i: a for i, a in enumerate(audios)}
    arrays["meta"] = np.array(json.dumps(zones_meta))
    path = os.path.join(OUT, name + ".npz")
    np.savez_compressed(path, **arrays)
    return os.path.getsize(path)


def main(sf_path):
    os.makedirs(OUT, exist_ok=True)
    sf = SoundFont(sf_path)
    total = 0

    for name, (bank, prog, max_sec) in MELODIC.items():
        zones = pick_zones(sf.preset_zones(bank, prog))
        metas, audios = [], []
        for z in zones:
            audio, sr, ls, le = sf.sample_audio(z["sample_index"])
            looped = z["loop_mode"] in (1, 3) and 0 <= ls < le <= len(audio) and le - ls > 32
            limit = int(max_sec * sr)
            if looped:
                cut = min(len(audio), le + 64)
                if cut > limit:  # döngü çok sonda: döngüyü kısaltmadan sınırla
                    cut = len(audio) if le + 64 > len(audio) else le + 64
            else:
                cut = min(len(audio), limit)
                fade = min(int(0.05 * sr), cut)
                audio = audio.copy()
                audio[cut - fade:cut] *= np.linspace(1, 0, fade)
            audio = audio[:cut]
            gain = 10 ** (-z["attenuation_cb"] / 200.0)
            metas.append({
                "key_lo": z["key_lo"], "key_hi": z["key_hi"],
                "root": z["root"], "tune_cents": z["tune_cents"],
                "sr": sr, "loop": [int(ls), int(le)] if looped else None,
                "peak": float(np.abs(audio).max() * gain),
            })
            audios.append(to_int16(audio))
        size = save(name, metas, audios)
        total += size
        print("%-9s %2d bölge  %6.0f KB  (%s)" % (name, len(metas), size / 1024, zones[0]["inst"]))

    drum_zones = sf.preset_zones(128, 0)
    for name, key in DRUMS.items():
        zs = [z for z in drum_zones if z["key_lo"] <= key <= z["key_hi"] and z["vel_lo"] <= 100 <= z["vel_hi"]]
        z = zs[0]
        audio, sr, _, _ = sf.sample_audio(z["sample_index"])
        cut = min(len(audio), int(1.5 * sr))
        audio = audio[:cut].copy()
        fade = min(int(0.03 * sr), cut)
        audio[cut - fade:] *= np.linspace(1, 0, fade)
        meta = [{"key_lo": key, "key_hi": key, "root": key, "tune_cents": 0, "sr": sr, "loop": None,
                 "peak": float(np.abs(audio).max())}]
        size = save("davul_" + name, meta, [to_int16(audio)])
        total += size
        print("%-16s %6.0f KB  (%s)" % ("davul_" + name, size / 1024, z["sample"]))
    print("toplam %.1f MB" % (total / 1048576))


if __name__ == "__main__":
    main(sys.argv[1])
