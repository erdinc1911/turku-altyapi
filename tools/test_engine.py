"""Motor testi (masaüstünde çalışır): sentetik "saz + vokal" kaydı üretir, analiz eder,
bilinen değerlerle karşılaştırır ve stemleri yazar.

Kullanım:  python tools/test_engine.py [cikti_klasoru]
"""

import json
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "src", "main", "python"))

from altyapi import pipeline, wavio  # noqa: E402


def make_song(path, bpm=100.0, tonic_midi=57, tuning=20.0, seconds=60.0, sr=44100, seed=3):
    """La (57) karar sesli, uşşak benzeri dizide, tezene vuruşlu bağlama + vokal."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    y = np.zeros(n, dtype=np.float64)
    beat = 60.0 / bpm
    scale = [0, 2, 3, 5, 7, 8, 10, 12]  # yaklaşık; koma perdeleri ihmal
    # basit ezgi: her ölçü 2 vuruş, sekizlik tezene
    melody = []
    deg = 4
    for bar in range(int(seconds / (2 * beat))):
        for e in range(4):
            if rng.random() < 0.6:
                deg = int(np.clip(deg + rng.choice([-1, 1, -2, 2]), 0, len(scale) - 1))
            melody.append(scale[deg])
    melody[-8:] = [3, 2, 2, 0, 0, 0, 0, 0]  # karara iniş

    def hz(m):
        return 440 * 2 ** ((m - 69) / 12 + tuning / 1200)

    t8 = beat / 2
    for i, iv in enumerate(melody):
        start = int(i * t8 * sr)
        ln = int(0.6 * sr)
        t = np.arange(ln) / sr
        f = hz(tonic_midi + 12 + iv)
        accent = 1.0 if i % 4 == 0 else (0.75 if i % 2 == 0 else 0.5)
        pluck = sum(np.sin(2 * np.pi * k * f * t) / k for k in range(1, 6)) * np.exp(-t / 0.18)
        pluck += 0.3 * np.sin(2 * np.pi * hz(tonic_midi) * t) * np.exp(-t / 0.3)  # dem teli
        click = rng.standard_normal(ln) * np.exp(-t / 0.002) * 0.4
        seg = (pluck + click) * accent * 0.25
        end = min(n, start + ln)
        y[start:end] += seg[: end - start]
    # vokal: ezgiyi bir oktav yukarıda, vibratolu ve legato söyler (her vuruşta bir hece)
    for i in range(0, len(melody), 2):
        start = int(i * t8 * sr)
        ln = int(2 * t8 * sr)
        t = np.arange(ln) / sr
        f = hz(tonic_midi + 12 + melody[i])
        ph = 2 * np.pi * np.cumsum(f * (1 + 0.006 * np.sin(2 * np.pi * 5.5 * t))) / sr
        v = (np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)) * np.minimum(1, t / 0.05) * 0.12
        end = min(n, start + ln)
        y[start:end] += v[: end - start]
    y /= np.abs(y).max() * 1.1
    stereo = np.stack([y, y * 0.95], axis=1)
    wavio.write_wav(path, stereo, sr)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="altyapi_")
    os.makedirs(out, exist_ok=True)
    make_song(os.path.join(out, "source.wav"))

    log = []
    t0 = time.time()
    summary = json.loads(pipeline.run_full(out, json.dumps({"density": "dengeli"}),
                                           lambda p, m: log.append((p, m))))
    full_s = time.time() - t0
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("toplam süre: %.1f sn, ilerleme adımı: %d, son: %s" % (full_s, len(log), log[-1]))

    errors = []
    if not (95 <= summary["bpm"] <= 105):
        errors.append("tempo beklenen ~100, bulunan %s" % summary["bpm"])
    if summary["tonic_name"] != "La":
        errors.append("karar beklenen La, bulunan %s" % summary["tonic_name"])
    if abs(summary["tuning_cents"] - 20) > 8:
        errors.append("akort beklenen +20, bulunan %s" % summary["tuning_cents"])
    keys = [s["key"] for s in summary["stems"]]
    if keys != ["bas", "bendir", "org", "cura", "ney"]:
        errors.append("varsayılan stemler beklenenden farklı: %s" % keys)
    if any(b < a for a, b in zip([p for p, _ in log], [p for p, _ in log][1:])):
        errors.append("ilerleme geri gitti")

    with open(os.path.join(out, "arrangement.json")) as f:
        bas = json.load(f)["events"]["bas"]
    gaps = np.diff([e["t"] for e in bas])
    print("bas: %d nota, en kısa aralık %.2f sn" % (len(bas), gaps.min()))
    if gaps.min() < 0.5:
        errors.append("bas notaları çok sık (%.2f sn)" % gaps.min())

    t1 = time.time()
    s2 = json.loads(pipeline.run_arrange(out, json.dumps({"density": "zengin", "tempo_factor": 2.0,
                                                          "instruments": ["bas", "yaylilar", "darbuka"]})))
    print("yeniden üretim: %.1f sn, bpm %s, stemler %s" % (time.time() - t1, s2["bpm"],
                                                          [s["key"] for s in s2["stems"]]))
    if not (190 <= s2["bpm"] <= 210):
        errors.append("tempo x2 hatalı: %s" % s2["bpm"])
    if os.path.exists(os.path.join(out, "ney.wav")):
        errors.append("seçilmeyen ney stemi silinmedi")

    ref_json = os.path.join(out, "ref_profile.json")
    pipeline.reference_profile(os.path.join(out, "source.wav"), ref_json)
    st = json.loads(pipeline.run_arrange(out, json.dumps({"density": "dengeli", "ref_profile": ref_json})))
    print("ses rengi:", st.get("tone"))
    if not st.get("tone"):
        errors.append("referans ses rengi uygulanmadı")

    v = json.loads(pipeline.run_variant(out, json.dumps({"density": "dengeli"})))
    print(v["variant_label"], [s["file"] for s in v["stems"]])
    if not all(s["file"].startswith("b/") for s in v["stems"]):
        errors.append("B versiyonu alt klasöre yazılmadı")
    pipeline.promote_variant(out)
    if os.path.isdir(os.path.join(out, "b")):
        errors.append("B seçilince alt klasör kaldırılmadı")

    print("klasör:", out)
    if errors:
        print("HATALAR:\n- " + "\n- ".join(errors))
        sys.exit(1)
    print("TÜM KONTROLLER GEÇTİ")


if __name__ == "__main__":
    main()
