"""Ses üretimi.

- Örnekleyici: gerçek çalgı kayıtlarını (FluidR3 Mono, MIT) perde kaydırarak çalar
  (bas, yaylılar, ney, bendir/darbuka vuruşları).
- Org: çekmeceli (drawbar) org katkısal sentezi.
- Cura: Karplus-Strong tel modeli (mızrapla çalınan metal tel).

Tüm perdeler kaydın ölçülen akort sapmasına (sent) göre ayarlanır.
"""

import io
import json
import os
import pkgutil

import numpy as np

TABLE_LEN = 4096


def midi_hz(m, tuning_cents=0.0):
    return 440.0 * 2.0 ** ((m - 69) / 12.0 + tuning_cents / 1200.0)


def _add(out, start, sig):
    if start >= len(out) or len(sig) == 0:
        return
    if start < 0:
        sig = sig[-start:]
        start = 0
    end = min(len(out), start + len(sig))
    out[start:end] += sig[: end - start]


def _release_env(n, sr, dur, attack, release):
    t = np.arange(n, dtype=np.float32) / sr
    env = np.minimum(1.0, t / max(attack, 1e-4)) if attack > 0 else np.ones(n, dtype=np.float32)
    rel = np.exp(-np.maximum(t - dur, 0.0) / max(release / 4.0, 1e-3))
    return (env * rel).astype(np.float32)


# ---------------------------------------------------------------- örnekleyici

_BANK_CACHE = {}


def _load_bank(name):
    if name in _BANK_CACHE:
        return _BANK_CACHE[name]
    raw = None
    try:  # Android'de (Chaquopy) dosyalar uygulama paketinin içinden okunur
        raw = pkgutil.get_data("altyapi", "samples/%s.npz" % name)
    except Exception:
        raw = None
    if raw is None:
        with open(os.path.join(os.path.dirname(__file__), "samples", name + ".npz"), "rb") as f:
            raw = f.read()
    d = np.load(io.BytesIO(raw))
    meta = json.loads(str(d["meta"]))
    zones = []
    for i, z in enumerate(meta):
        z = dict(z)
        z["audio"] = d["a%d" % i].astype(np.float32) / 32767.0
        zones.append(z)
    _BANK_CACHE[name] = zones
    return zones


def _pick_zone(zones, midi):
    for z in zones:
        if z["key_lo"] <= midi <= z["key_hi"]:
            return z
    return min(zones, key=lambda z: abs(z["root"] - midi))


def sample_note(bank, midi, dur, vel, sr, tuning=0.0, attack=0.004, release=0.12,
                decay=None, max_len=None, shift_cents=0.0):
    zones = _load_bank(bank)
    z = _pick_zone(zones, int(round(midi)))
    shift = (midi - z["root"]) * 100.0 + z["tune_cents"] + tuning + shift_cents
    ratio = 2.0 ** (shift / 1200.0) * z["sr"] / sr
    audio = z["audio"]
    n = int((dur + release) * sr)
    if max_len:
        n = min(n, int(max_len * sr))
    pos = np.arange(n, dtype=np.float64) * ratio
    loop = z["loop"]
    if loop:
        ls, le = loop
        span = le - ls
        over = pos >= le
        pos[over] = ls + np.mod(pos[over] - ls, span)
    else:
        n_ok = int(np.searchsorted(pos, len(audio) - 1))
        pos = pos[:n_ok]
        n = n_ok
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    i0 = pos.astype(np.int64)
    fr = (pos - i0).astype(np.float32)
    i1 = np.minimum(i0 + 1, len(audio) - 1)
    sig = audio[i0] + (audio[i1] - audio[i0]) * fr
    env = _release_env(n, sr, dur, attack, release)
    if decay:
        env *= np.exp(-np.arange(n, dtype=np.float32) / (decay * sr))
    gain = (vel ** 1.4) * 0.5  # örnekler paketlenirken tepe değerine göre eşitlendi
    return (sig * env * gain).astype(np.float32)


def render_sampled(events, n, sr, tuning, bank, **kw):
    out = np.zeros(n, dtype=np.float32)
    for e in events:
        _add(out, int(e["t"] * sr), sample_note(bank, e["midi"], e["dur"], e["vel"], sr, tuning, **kw))
    return out


# ---------------------------------------------------------------- vurmalılar

# tür -> [(örnek paketi, perde kaydırma (yarım ses), kazanç, gecikme sn)]
PERC_KITS = {
    # Bendir: gövde (derin düm) + kiriş vızıltısı + parmak; VSCO 2 CE (CC0) kayıtları
    "bendir": {
        "dum": [("bendir_govde", 1.0, 1.0, 0.0), ("bendir_vizilti", -3.0, 0.22, 0.004)],
        "tek": [("bendir_vizilti", 1.0, 0.75, 0.0), ("bendir_parmak", 0.0, 0.35, 0.0)],
        "ka": [("bendir_parmak", 0.0, 0.5, 0.0), ("bendir_vizilti", 2.0, 0.12, 0.002)],
    },
    "darbuka": {
        "dum": [("davul_tumba", -1.0, 0.95, 0.0)],
        "tek": [("davul_timbale", 2.0, 0.65, 0.0)],
        "ka": [("davul_bongo_rim", 0.0, 0.55, 0.0)],
    },
}


def _perc_hit(bank, variant, semis, vel, sr):
    zones = _load_bank(bank)
    z = zones[variant % len(zones)]
    ratio = 2.0 ** (semis / 12.0) * z["sr"] / sr
    audio = z["audio"]
    n = int(min(len(audio) / ratio, 1.4 * sr))
    pos = np.arange(n, dtype=np.float64) * ratio
    i0 = pos.astype(np.int64)
    i1 = np.minimum(i0 + 1, len(audio) - 1)
    fr = (pos - i0).astype(np.float32)
    sig = audio[i0] + (audio[i1] - audio[i0]) * fr
    # yumuşak vuruşlar biraz daha kısa ve koyu: hıza bağlı hafif sönüm
    sig *= np.exp(-np.arange(n, dtype=np.float32) / ((0.25 + 0.5 * vel) * sr))
    return (sig * (vel ** 1.3) * 0.6).astype(np.float32)


def render_perc(events, n, sr, kit, seed=1):
    rng = np.random.default_rng(seed + 7)
    out = np.zeros(n, dtype=np.float32)
    layers = PERC_KITS[kit]
    for i, e in enumerate(events):
        start = int(e["t"] * sr)
        variant = i + int(rng.integers(0, 2))  # aynı vuruşun art arda tekrarını kır
        for bank, semis, gain, delay in layers[e["kind"]]:
            sig = _perc_hit(bank, variant, semis + rng.normal(0, 0.08), e["vel"], sr)
            _add(out, start + int(delay * sr), sig * gain)
    return out


# ---------------------------------------------------------------- org (çekmeceli)

# 16', 5 1/3', 8', 4', 2 2/3', 2', 1 3/5', 1 1/3', 1'
DRAWBAR_RATIOS = [0.5, 1.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
DRAWBARS = [6, 2, 8, 5, 0, 3, 0, 0, 1]


def _organ_table():
    x = np.arange(TABLE_LEN) / TABLE_LEN
    tab = np.zeros(TABLE_LEN)
    # tablo 16' (yarım frekans) temel alınarak kurulur: oranlar x2
    for r, lvl in zip(DRAWBAR_RATIOS, DRAWBARS):
        if lvl:
            tab += (lvl / 8.0) * np.sin(2 * np.pi * (r * 2) * x)
    return (tab / np.abs(tab).max()).astype(np.float32)


def _table_osc(table, f, n, sr, rng, vib_rate=6.5, vib_depth=0.0018):
    t = np.arange(n, dtype=np.float64) / sr
    vib = 1.0 + vib_depth * np.sin(2 * np.pi * vib_rate * t + rng.uniform(0, 6.28))
    phase = (np.cumsum(f * vib * TABLE_LEN / sr) + rng.uniform(0, TABLE_LEN)) % TABLE_LEN
    i0 = phase.astype(np.int64)
    fr = (phase - i0).astype(np.float32)
    a = table[i0]
    return a + (table[(i0 + 1) % TABLE_LEN] - a) * fr


def render_organ(events, n, sr, tuning, seed=1):
    rng = np.random.default_rng(seed + 21)
    table = _organ_table()
    out = np.zeros(n, dtype=np.float32)
    for e in events:
        f16 = midi_hz(e["midi"], tuning) / 2.0  # tablo 16' tabanlı
        m = int((e["dur"] + 0.08) * sr)
        # iki hafif ayrık ses: dönen hoparlör (leslie) benzeri koro
        sig = 0.6 * _table_osc(table, f16, m, sr, rng) + 0.4 * _table_osc(table, f16 * 1.0015, m, sr, rng, 0.8, 0.0)
        t = np.arange(m, dtype=np.float32) / sr
        trem = 1.0 + 0.08 * np.sin(2 * np.pi * 5.8 * t)
        click = np.zeros(m, dtype=np.float32)
        k = min(m, int(0.004 * sr))
        click[:k] = rng.standard_normal(k) * 0.05
        env = _release_env(m, sr, e["dur"], 0.008, 0.08)
        _add(out, int(e["t"] * sr), ((sig * trem + click) * env * 0.16 * e["vel"]).astype(np.float32))
    return out


# ---------------------------------------------------------------- cura (tel modeli)

def ks_pluck(f, dur, vel, sr, rng, brightness=0.28, sustain=0.998):
    """Karplus-Strong: blok blok vektörel hesap. Perde, sonradan ince yeniden örneklemeyle düzeltilir."""
    b = brightness
    N = max(8, int(sr / f - b))
    f_actual = sr / (N + b)  # ortalama süzgeci b örnek gecikme ekler
    ratio = f / f_actual
    n_out = int(min(dur + 0.06, 4.0) * sr)
    total = int(n_out * ratio) + N + 3
    y = np.zeros(total, dtype=np.float32)
    exc = rng.uniform(-1, 1, N).astype(np.float32)
    exc = exc - np.roll(exc, max(1, int(N * 0.18)))  # mızrap konumu (köprüye yakın: parlak)
    y[:N] = exc
    for k in range(N, total, N):
        m = min(N, total - k)
        prev = y[k - N: k - N + m]
        if k - N - 1 >= 0:
            prev1 = y[k - N - 1: k - N - 1 + m]
        else:
            prev1 = np.concatenate([[0.0], prev[:-1]]).astype(np.float32)
        y[k:k + m] = sustain * ((1 - b) * prev + b * prev1)
    pos = np.arange(n_out) * ratio
    pos = pos[pos < total - 1]
    i0 = pos.astype(np.int64)
    fr = (pos - i0).astype(np.float32)
    sig = y[i0] + (y[i0 + 1] - y[i0]) * fr
    env = _release_env(len(sig), sr, dur, 0.0, 0.05)
    return (sig * env * 0.35 * vel).astype(np.float32)


def render_cura(events, n, sr, tuning, seed=1):
    rng = np.random.default_rng(seed + 33)
    out = np.zeros(n, dtype=np.float32)
    for e in events:
        f = midi_hz(e["midi"], tuning)
        # cura çift telli çalınır: hafif akort farkıyla iki tel
        s1 = ks_pluck(f, e["dur"], e["vel"], sr, rng)
        s2 = ks_pluck(f * 1.0012, e["dur"], e["vel"] * 0.7, sr, rng)
        _add(out, int(e["t"] * sr), s1)
        _add(out, int(e["t"] * sr) + int(0.002 * sr), s2)
    return out


def safe_peak(x, ceiling=0.95):
    p = float(np.abs(x).max()) if len(x) else 0.0
    return x * (ceiling / p) if p > ceiling else x


# ---------------------------------------------------------------- yankı (oda)

def _room_ir(sr, seconds, seed):
    rng = np.random.default_rng(seed + 99)
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * t / seconds)
    # yüksek frekanslar daha çabuk sönsün: kaba alçak geçiren (hareketli ortalama)
    k = 6
    ir = np.convolve(ir, np.ones(k, dtype=np.float32) / k, mode="same")
    ir[: int(0.012 * sr)] = 0.0  # ilk yansımadan önceki boşluk (pre-delay)
    return ir / np.sqrt(np.sum(ir ** 2))


def reverb(x, sr, wet=0.3, seconds=1.8, seed=1):
    """FFT tabanlı, parça parça (overlap-add) oda yankısı. Kuru sesle karıştırılmış döner."""
    if wet <= 0 or len(x) == 0:
        return x
    ir = _room_ir(sr, seconds, seed)
    block = 1 << 16
    nfft = 1 << int(np.ceil(np.log2(block + len(ir) - 1)))
    H = np.fft.rfft(ir, nfft)
    out = np.zeros(len(x) + len(ir), dtype=np.float32)
    for start in range(0, len(x), block):
        seg = x[start:start + block]
        y = np.fft.irfft(np.fft.rfft(seg, nfft) * H, nfft)[: len(seg) + len(ir) - 1]
        out[start:start + len(y)] += y.astype(np.float32)
    wet_sig = out[: len(x)] * 0.5
    return ((1.0 - wet * 0.5) * x + wet * wet_sig).astype(np.float32)
