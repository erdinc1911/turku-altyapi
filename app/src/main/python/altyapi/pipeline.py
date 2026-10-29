"""Uygulamanın çağırdığı giriş noktaları.

Proje klasörü yapısı:
    source.wav      kaynak kayıt (Android tarafı çözüp yazar)
    analysis.json   analiz sonucu (bir kez hesaplanır)
    settings.json   son kullanılan düzenleme ayarları
    arrangement.json nota olayları
    bas.wav, perkusyon.wav, pad.wav   üretilen stemler

`run_full` analiz + üretim yapar. `run_arrange` analizi yeniden yapmadan
yalnız düzenleme ve sesi yeniden üretir (ayar değişikliklerinde hızlıdır).
"""

import json
import os
import time

import numpy as np

from . import analysis, arrange, synth, wavio

STEMS = [
    ("bas", "Bas"),
    ("bendir", "Bendir"),
    ("darbuka", "Darbuka"),
    ("org", "Org"),
    ("yaylilar", "Yaylılar"),
    ("cura", "Cura"),
    ("ney", "Ney"),
]


class _Progress:
    def __init__(self, cb, lo=0, hi=100):
        self.cb, self.lo, self.hi = cb, lo, hi

    def __call__(self, pct, msg):
        if self.cb is None:
            return
        val = int(self.lo + (self.hi - self.lo) * pct / 100.0)
        try:
            self.cb.report(val, msg)
        except AttributeError:
            self.cb(val, msg)


# çalgı başına oda yankısı oranı (ney doğal olarak odada sönerek kaybolur)
REVERB = {"ney": 0.45, "yaylilar": 0.3, "org": 0.2, "cura": 0.18, "bendir": 0.12, "darbuka": 0.1, "bas": 0.0}


def _path(project_dir, name):
    return os.path.join(project_dir, name)


def _load_mono(project_dir):
    x, sr = wavio.read_wav(_path(project_dir, "source.wav"))
    return wavio.to_mono(x), sr, len(x)


def analyze(project_dir, progress=None):
    p = _Progress(progress) if not isinstance(progress, _Progress) else progress
    t0 = time.time()
    mono, sr, _ = _load_mono(project_dir)
    an = analysis.analyze(mono, sr, p)
    an["analysis_seconds"] = round(time.time() - t0, 2)
    with open(_path(project_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(an, f)
    return an


def _render(project_dir, an, settings, progress):
    t0 = time.time()
    progress(5, "Düzenleme hazırlanıyor")
    arr = arrange.arrange(an, settings)
    with open(_path(project_dir, "arrangement.json"), "w", encoding="utf-8") as f:
        json.dump(arr, f)

    sr = int(an["sample_rate"])
    tail = 3.0  # bitiş notası için pay
    n = int((an["duration"] + tail) * sr)
    tuning = float(an["tuning_cents"])
    seed = int(settings.get("seed", 1))
    ev = arr.get("events", {})

    produced = []
    renderers = {
        "bas": (lambda e: synth.render_sampled(e, n, sr, tuning, "bas", attack=0.004, release=0.08),
                "Bas çalınıyor"),
        "bendir": (lambda e: synth.render_perc(e, n, sr, "bendir", seed), "Bendir çalınıyor"),
        "darbuka": (lambda e: synth.render_perc(e, n, sr, "darbuka", seed), "Darbuka çalınıyor"),
        "org": (lambda e: synth.render_organ(e, n, sr, tuning, seed), "Org çalınıyor"),
        "yaylilar": (lambda e: synth.render_sampled(e, n, sr, tuning, "yaylilar", attack=0.35, release=0.6),
                     "Yaylılar çalınıyor"),
        "cura": (lambda e: synth.render_cura(e, n, sr, tuning, seed), "Cura çalınıyor"),
        "ney": (lambda e: synth.render_sampled(e, n, sr, tuning, "ney", attack=0.08, release=0.9),
                "Ney çalınıyor"),
    }
    keys = [k for k, _ in STEMS]
    for i, key in enumerate(keys):
        fn, msg = renderers[key]
        path = _path(project_dir, key + ".wav")
        if ev.get(key):
            progress(10 + 85 * i / len(keys), msg)
            audio = fn(ev[key])
            if REVERB.get(key):
                audio = synth.reverb(audio, sr, REVERB[key], seed=seed)
            audio = synth.safe_peak(audio)
            wavio.write_wav(path, audio, sr)
            produced.append(key)
        elif os.path.exists(path):
            os.remove(path)

    progress(100, "Hazır")
    return arr, produced, round(time.time() - t0, 2)


def _summary(an, arr, produced, render_seconds):
    names = dict(STEMS)
    tonic = arr.get("tonic_pc", an["tonic_pc"])
    return json.dumps({
        "ok": arr.get("bars", 0) > 0,
        "info": arr.get("info", ""),
        "bpm": arr.get("bpm", an["bpm"]),
        "beats_per_bar": arr.get("beats_per_bar", an["beats_per_bar"]),
        "bars": arr.get("bars", 0),
        "tonic_pc": tonic,
        "tonic_name": analysis.NOTE_NAMES[tonic],
        "quality": an["quality"],
        "tuning_cents": an["tuning_cents"],
        "duration": round(an["duration"], 2),
        "analysis_seconds": an.get("analysis_seconds"),
        "render_seconds": render_seconds,
        "stems": [{"key": k, "name": names[k], "file": k + ".wav"} for k in produced],
    }, ensure_ascii=False)


def run_full(project_dir, settings_json, progress=None):
    settings = json.loads(settings_json or "{}")
    an = analyze(project_dir, _Progress(progress, 0, 60))
    with open(_path(project_dir, "settings.json"), "w", encoding="utf-8") as f:
        f.write(settings_json or "{}")
    arr, produced, secs = _render(project_dir, an, settings, _Progress(progress, 60, 100))
    return _summary(an, arr, produced, secs)


def run_arrange(project_dir, settings_json, progress=None):
    settings = json.loads(settings_json or "{}")
    with open(_path(project_dir, "analysis.json"), encoding="utf-8") as f:
        an = json.load(f)
    with open(_path(project_dir, "settings.json"), "w", encoding="utf-8") as f:
        f.write(settings_json or "{}")
    arr, produced, secs = _render(project_dir, an, settings, _Progress(progress, 0, 100))
    return _summary(an, arr, produced, secs)
