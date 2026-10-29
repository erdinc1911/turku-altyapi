"""Basit sentezleyiciler (ilk sürüm). İleride örnek tabanlı (SoundFont/SFZ) motorla değişecek.

Tüm frekanslar kaydın ölçülen akort sapmasına (sent) göre ayarlanır;
böylece altyapı bağlamanın gerçek akortuna uyar.
"""

import numpy as np

TABLE_LEN = 4096


def midi_hz(m, tuning_cents=0.0):
    return 440.0 * 2.0 ** ((m - 69) / 12.0 + tuning_cents / 1200.0)


def _add(out, start, sig):
    if start >= len(out):
        return
    if start < 0:
        sig = sig[-start:]
        start = 0
    end = min(len(out), start + len(sig))
    out[start:end] += sig[: end - start]


# ---------------------------------------------------------------- bas

def bass_note(f, dur, vel, sr):
    rel = 0.08
    n = int((dur + rel) * sr)
    t = np.arange(n) / sr
    w = 2 * np.pi * f * t
    tone = np.sin(w) + 0.45 * np.sin(2 * w) + 0.18 * np.sin(3 * w) + 0.07 * np.sin(4 * w)
    att = np.minimum(1.0, t / 0.006)
    body = np.exp(-t / 1.1)
    release = np.where(t > dur, np.exp(-(t - dur) / 0.03), 1.0)
    pluck = 1.0 + 0.6 * np.exp(-t / 0.04)  # tel çekme vurgusu
    sig = np.tanh(1.3 * tone * att * body * release * pluck) * 0.55 * vel
    return sig.astype(np.float32)


def render_bass(events, n, sr, tuning):
    out = np.zeros(n, dtype=np.float32)
    for e in events:
        _add(out, int(e["t"] * sr), bass_note(midi_hz(e["midi"], tuning), e["dur"], e["vel"], sr))
    return out


# ---------------------------------------------------------------- perküsyon (darbuka benzeri)

def _dum(sr, rng):
    n = int(0.45 * sr)
    t = np.arange(n) / sr
    f = 58 + 45 * np.exp(-t / 0.025)
    phase = 2 * np.pi * np.cumsum(f) / sr
    body = np.sin(phase) * np.exp(-t / 0.22)
    click = rng.standard_normal(n) * np.exp(-t / 0.003) * 0.3
    return body + click


def _tek(sr, rng, ring=520.0, decay=0.045, amp=1.0):
    n = int(0.25 * sr)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n)
    noise = np.concatenate([[0.0], np.diff(noise)])  # kaba yüksek geçiren
    sig = noise * np.exp(-t / 0.012) * 0.35 + np.sin(2 * np.pi * ring * t) * np.exp(-t / decay) * 0.6
    return sig * amp


def render_perc(events, n, sr, seed=1):
    rng = np.random.default_rng(seed + 7)
    bank = {
        "dum": [_dum(sr, rng) for _ in range(3)],
        "tek": [_tek(sr, rng) for _ in range(3)],
        "ka": [_tek(sr, rng, ring=700.0, decay=0.025, amp=0.6) for _ in range(3)],
    }
    out = np.zeros(n, dtype=np.float32)
    for i, e in enumerate(events):
        sample = bank[e["kind"]][i % 3]  # tekrar eden vuruşlarda farklı örnek (round-robin)
        _add(out, int(e["t"] * sr), (sample * 0.7 * e["vel"]).astype(np.float32))
    return out


# ---------------------------------------------------------------- yaylı pad

def _saw_table():
    x = np.arange(TABLE_LEN) / TABLE_LEN
    tab = np.zeros(TABLE_LEN)
    for k in range(1, 24):
        tab += np.sin(2 * np.pi * k * x) / k * np.exp(-k / 7.0)
    return (tab / np.abs(tab).max()).astype(np.float32)


def pad_note(f, dur, vel, sr, table, rng):
    att, rel = 0.7, 0.9
    n = int((dur + rel) * sr)
    t = np.arange(n, dtype=np.float64) / sr
    sig = np.zeros(n, dtype=np.float32)
    for detune in (-8.0, 0.0, 7.0):
        vib = 1.0 + 0.0025 * np.sin(2 * np.pi * rng.uniform(4.2, 5.2) * t + rng.uniform(0, 6.28))
        inc = f * 2 ** (detune / 1200.0) * vib * TABLE_LEN / sr
        phase = (np.cumsum(inc) + rng.uniform(0, TABLE_LEN)) % TABLE_LEN
        i0 = phase.astype(np.int64)
        frac = (phase - i0).astype(np.float32)
        a = table[i0]
        b = table[(i0 + 1) % TABLE_LEN]
        sig += a + (b - a) * frac
    env = np.minimum(1.0, t / att) * np.where(t > dur, np.exp(-(t - dur) / (rel / 3)), 1.0)
    return (sig / 3.0 * env * 0.22 * vel).astype(np.float32)


def render_pad(events, n, sr, tuning, seed=1):
    rng = np.random.default_rng(seed + 13)
    table = _saw_table()
    out = np.zeros(n, dtype=np.float32)
    for e in events:
        _add(out, int(e["t"] * sr), pad_note(midi_hz(e["midi"], tuning), e["dur"], e["vel"], sr, table, rng))
    return out


def safe_peak(x, ceiling=0.95):
    p = float(np.abs(x).max()) if len(x) else 0.0
    return x * (ceiling / p) if p > ceiling else x
