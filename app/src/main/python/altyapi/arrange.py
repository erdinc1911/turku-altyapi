"""Kural tabanlı düzenleme: analiz sonucundan çalgı başına nota olayları üretir.

İlk sürümün müzikal yaklaşımı (Türk halk müziği için):
- Bas: karar sesi ve beşlisi üzerinde "dem" tutar, usul vurgularını destekler.
- Perküsyon: darbuka benzeri düm / tek / ka kalıpları.
- Yaylı pad: karar + beşli + oktav (yanlış üçlü riskini almamak için akorsuz "dem");
  "zengin" yoğunlukta tespit edilen üçlü de eklenir.
Bölüm dinamiği kaynak kaydın vuruş enerjisinden türetilir.
"""

import numpy as np

from .analysis import downbeat_phase


def apply_tempo_factor(beats, strength, factor):
    """Tempo x2: vuruşların arasına ara vuruş ekler. Tempo /2: güçlü olan yarıyı tutar."""
    beats = np.asarray(beats, dtype=float)
    strength = np.asarray(strength, dtype=float)
    while factor >= 2 and len(beats) >= 2:
        mids = (beats[:-1] + beats[1:]) / 2
        new = np.empty(len(beats) + len(mids))
        new[0::2], new[1::2] = beats, mids
        s = np.empty(len(new))
        s[0::2], s[1::2] = strength, strength.min() if len(strength) else 0
        beats, strength = new, s
        factor /= 2
    while factor <= 0.5 and len(beats) >= 4:
        k = 0 if strength[0::2].sum() >= strength[1::2].sum() else 1
        beats, strength = beats[k::2], strength[k::2]
        factor *= 2
    return beats, strength


class BeatClock:
    """Kesirli vuruş konumunu saniyeye çevirir (tempo değişimlerini izler)."""

    def __init__(self, beats):
        self.beats = np.asarray(beats, dtype=float)
        self.period = float(np.median(np.diff(self.beats))) if len(self.beats) > 1 else 0.5

    def time(self, pos):
        n = len(self.beats)
        if pos <= 0:
            return self.beats[0] + pos * self.period
        if pos >= n - 1:
            return self.beats[-1] + (pos - (n - 1)) * self.period
        i = int(pos)
        frac = pos - i
        return self.beats[i] + frac * (self.beats[i + 1] - self.beats[i])


# kalıplar: (vuruş içindeki konum, süre [vuruş], aralık [karar=0, beşli=7, oktav=12], hız)
# Türküde bas seyrek çalar: ölçü başında karar sesini uzun tutar (dem),
# yoğunluk arttıkça yalnız yarım ölçüde beşliyle destekler.
BASS_PATTERNS = {
    2: {
        "sade": [(0.0, 2.0, 0, 0.9)],
        "dengeli": [(0.0, 2.0, 0, 0.95)],
        "zengin": [(0.0, 1.0, 0, 1.0), (1.0, 1.0, 7, 0.8)],
    },
    3: {
        "sade": [(0.0, 3.0, 0, 0.9)],
        "dengeli": [(0.0, 3.0, 0, 0.95)],
        "zengin": [(0.0, 2.0, 0, 1.0), (2.0, 1.0, 7, 0.8)],
    },
    4: {
        "sade": [(0.0, 4.0, 0, 0.9)],
        "dengeli": [(0.0, 2.0, 0, 0.95), (2.0, 2.0, 7, 0.8)],
        "zengin": [(0.0, 2.0, 0, 1.0), (2.0, 1.0, 7, 0.85), (3.0, 1.0, 0, 0.75)],
    },
}

# Bas notaları arasında en az bu kadar süre olur (hızlı tempoda sıklaşmayı önler).
BASS_MIN_GAP = 0.55

# (konum, tür, hız)
PERC_PATTERNS = {
    2: {
        "sade": [(0.0, "dum", 0.9), (1.0, "tek", 0.6)],
        "dengeli": [(0.0, "dum", 1.0), (1.0, "tek", 0.8), (1.5, "ka", 0.45)],
        "zengin": [(0.0, "dum", 1.0), (0.5, "ka", 0.45), (1.0, "tek", 0.85), (1.5, "ka", 0.5), (1.75, "ka", 0.35)],
    },
    3: {
        "sade": [(0.0, "dum", 0.9), (2.0, "tek", 0.6)],
        "dengeli": [(0.0, "dum", 1.0), (1.0, "tek", 0.7), (2.0, "tek", 0.75)],
        "zengin": [(0.0, "dum", 1.0), (1.0, "tek", 0.75), (1.5, "ka", 0.4), (2.0, "tek", 0.8), (2.5, "ka", 0.45)],
    },
    4: {
        "sade": [(0.0, "dum", 0.9), (2.0, "tek", 0.65)],
        "dengeli": [(0.0, "dum", 1.0), (1.0, "tek", 0.7), (1.5, "ka", 0.4), (2.0, "dum", 0.85), (3.0, "tek", 0.8)],
        "zengin": [(0.0, "dum", 1.0), (0.5, "ka", 0.4), (1.0, "tek", 0.75), (1.5, "ka", 0.45),
                   (2.0, "dum", 0.9), (2.5, "ka", 0.4), (3.0, "tek", 0.85), (3.5, "ka", 0.5)],
    },
}

DEFAULT_INSTRUMENTS = ["bas", "bendir", "org", "cura", "ney"]

LEVEL_PATTERN = {
    # (istenen yoğunluk, bölüm seviyesi) -> kullanılacak kalıp
    "sade": ["sade", "sade", "sade"],
    "dengeli": ["sade", "dengeli", "dengeli"],
    "zengin": ["dengeli", "zengin", "zengin"],
}


def _bar_levels(beat_energy, bars):
    e = np.asarray(beat_energy, dtype=float)
    bar_e = np.array([e[[i for i in bar if i < len(e)]].mean() if bar else 0.0 for bar in bars])
    if len(bar_e) == 0:
        return np.zeros(0, dtype=int)
    lo, hi = np.percentile(bar_e, 15), np.percentile(bar_e, 85)
    rel = np.clip((bar_e - lo) / (hi - lo + 1e-12), 0, 1)
    # 3 ölçülük medyan ile yumuşat
    sm = rel.copy()
    for i in range(len(rel)):
        sm[i] = np.median(rel[max(0, i - 1): i + 2])
    return np.digitize(sm, [0.33, 0.66])  # 0 düşük, 1 orta, 2 yüksek


def _bass_root(pc):
    root = 24 + pc
    return root + 12 if root < 31 else root  # yaklaşık Sol1 - Fa#2 aralığı


def _pad_root(pc):
    root = 48 + pc
    return root - 12 if root > 56 else root


# ---------------------------------------------------------------- armoni (bas yürüyüşü)

# bas kökü ile ezgi notası arasındaki aralığın uyum puanı (yarım ses -> puan)
FIT = {0: 1.0, 7: 0.7, 3: 0.55, 4: 0.55, 8: 0.3, 9: 0.3, 5: 0.25, 10: 0.25,
       2: -0.1, 1: -0.5, 11: -0.4, 6: -0.4}
ROOT_CANDIDATES = [0, 5, 7, 10, 3, 8, 2]  # karar, 4., 5., 7., 3., 6., 2. derece


def _scale_from_chroma(chroma, tonic):
    """Kaydın en çok kullanılan 7 perde sınıfı (karara göre), karar her zaman içinde."""
    c = np.asarray(chroma, dtype=float)
    rel = np.roll(c, -tonic)
    top = set(int(i) for i in np.argsort(rel)[::-1][:7])
    top.add(0)
    return top


def _bar_roots(bars, clock, melody, tonic, scale):
    """Her ölçü için ezgiye en uyan bas kökünü seçer (karara göre yarım ses)."""
    cands = [r for r in ROOT_CANDIDATES if r in scale] or [0]
    notes = np.array(melody, dtype=float) if melody else np.zeros((0, 3))
    roots, prev = [], 0
    for k, bar in enumerate(bars):
        t0, t1 = clock.time(bar[0]), clock.time(bar[-1] + 1)
        if len(notes):
            sel = (notes[:, 0] < t1) & (notes[:, 0] + notes[:, 1] > t0)
            seg = notes[sel]
        else:
            seg = notes
        if len(seg) == 0:
            roots.append(prev)
            continue
        w = np.minimum(seg[:, 0] + seg[:, 1], t1) - np.maximum(seg[:, 0], t0)
        pcs = (seg[:, 2].astype(int) - tonic) % 12
        best, best_s = prev, -9.0
        for r in cands:
            fit = sum(wi * FIT[(pc - r) % 12] for wi, pc in zip(w, pcs)) / max(w.sum(), 1e-6)
            bias = 0.25 if r == 0 else (0.05 if r == 7 else 0.0)
            if k % 4 == 3:
                bias += 0.15 if r == 0 else 0.0  # 4 ölçülük cümle sonu kararda dinlensin
            if r == prev:
                bias += 0.1  # gereksiz kök değişimini önle
            if fit + bias > best_s:
                best, best_s = r, fit + bias
        roots.append(best)
        prev = best
    return roots


def _approach(next_root, scale):
    """Bir sonraki köke alttan (dizi içinden) yaklaşan geçiş sesi."""
    for step in range(1, 4):
        if (next_root - step) % 12 in scale:
            return next_root - step
    return next_root - 1


# ---------------------------------------------------------------- bendir (iki el)

# Bendir kalıpları: sabit ve düzenli (ölçüden ölçüye değişmez), yalnız sekizlik vuruşlar.
# (vuruş içi konum, D=düm merkez / T=tek kenar / k=parmak, hız)
BENDIR_PATTERNS = {
    2: {"sade": [(0, "D", 1.0), (1, "T", 0.8)],
        "dengeli": [(0, "D", 1.0), (0.5, "k", 0.4), (1, "T", 0.85), (1.5, "k", 0.4)],
        "zengin": [(0, "D", 1.0), (0.5, "k", 0.45), (1, "T", 0.9), (1.5, "D", 0.6)]},
    3: {"sade": [(0, "D", 1.0), (1, "T", 0.75), (2, "T", 0.75)],
        "dengeli": [(0, "D", 1.0), (0.5, "k", 0.4), (1, "T", 0.8), (2, "T", 0.8), (2.5, "k", 0.4)],
        "zengin": [(0, "D", 1.0), (0.5, "k", 0.45), (1, "T", 0.85), (1.5, "k", 0.4), (2, "T", 0.85),
                   (2.5, "k", 0.45)]},
    4: {"sade": [(0, "D", 1.0), (2, "T", 0.8)],
        "dengeli": [(0, "D", 1.0), (0.5, "k", 0.4), (1, "T", 0.8), (2, "D", 0.85), (2.5, "k", 0.4),
                    (3, "T", 0.85)],
        "zengin": [(0, "D", 1.0), (0.5, "k", 0.45), (1, "T", 0.85), (1.5, "k", 0.4), (2, "D", 0.9),
                   (2.5, "k", 0.45), (3, "T", 0.9), (3.5, "k", 0.45)]},
}
_KIND = {"D": "dum", "T": "tek", "k": "ka"}
PERC_MIN_GAP = 0.14  # iki vuruş arasında en az bu kadar (hızlı tempoda sekizlikler düşer)


def _bendir_part(bars, levels, bpb, density, clock, rng):
    ev = []
    for k, (bar, level) in enumerate(zip(bars, levels)):
        style = LEVEL_PATTERN[density][level]
        pat = list(BENDIR_PATTERNS[bpb][style])
        b0 = bar[0]
        # 8 ölçüde bir, yalnız coşkulu bölümlerde, son vuruşta kısa ve sakin bir geçiş
        if k % 8 == 7 and level == 2 and k < len(bars) - 1:
            last = bpb - 1
            pat = [p for p in pat if p[0] < last] + [(last, "T", 0.8), (last + 0.5, "T", 0.65)]
        for pos, kind, v in pat:
            t = clock.time(b0 + pos)
            if ev and t - ev[-1]["t"] < PERC_MIN_GAP:
                continue
            vv = v * (0.85 + 0.08 * level) * rng.uniform(0.94, 1.06)
            ev.append({"t": t + rng.normal(0, 0.003), "kind": _KIND[kind], "vel": float(np.clip(vv, 0.05, 1.0))})
    ev.append({"t": clock.time(bars[-1][-1] + 1), "kind": "dum", "vel": 1.0})
    return ev


# Bas vuruş yerleri (vuruş içi konum): yoğunluğa göre
BASS_SLOTS = {
    2: {"sade": [0], "dengeli": [0, 1], "zengin": [0, 1, 1.5]},
    3: {"sade": [0], "dengeli": [0, 2], "zengin": [0, 1, 2]},
    4: {"sade": [0, 2], "dengeli": [0, 1, 2, 3], "zengin": [0, 1, 2, 3, 3.5]},
}
CONSONANT = {0: 1.0, 3: 0.8, 4: 0.8, 7: 0.9, 8: 0.5, 9: 0.5}  # ezgi notasına göre bas aralığı


def _melody_pc_at(notes, t):
    """t anında çalan (ya da en son çalmış) ezgi notasının perde sınıfı."""
    if len(notes) == 0:
        return None
    i = int(np.searchsorted(notes[:, 0], t, side="right")) - 1
    if i < 0:
        return None
    return int(notes[i, 2]) % 12


def _bass_line(bars, levels, roots, bpb, density, clock, melody, tonic, scale, rng, vel):
    """Ezgiye karşı yürüyen bas: ölçü başında kök, aradaki vuruşlarda ezgiyle uyumlu ve
    önceki notaya yakın (adım adım) dizi sesleri. Aynı notayı üst üste tekrarlamaktan kaçınır."""
    base = _bass_root(tonic)
    lo, hi = 28, 47
    pool = [m for m in range(lo, hi + 1) if (m - base) % 12 in scale]
    notes = np.array(melody, dtype=float) if melody else np.zeros((0, 3))
    ev, prev = [], base
    for k, (bar, level) in enumerate(zip(bars, levels)):
        style = LEVEL_PATTERN[density][level]
        slots = BASS_SLOTS[bpb][style]
        b0 = bar[0]
        root_pc = (base + roots[k]) % 12
        for j, pos in enumerate(slots):
            t0 = clock.time(b0 + pos)
            nxt = slots[j + 1] if j + 1 < len(slots) else bpb
            t1 = clock.time(b0 + nxt)
            if ev and t0 - ev[-1]["t"] < BASS_MIN_GAP:
                continue
            mel = _melody_pc_at(notes, t0 + 0.05)
            best, best_s = prev, -9e9
            for m in pool:
                s_ = -0.12 * abs(m - prev)
                if pos == 0:
                    s_ += 2.0 if m % 12 == root_pc else -1.0  # ölçü başında kök
                elif mel is not None:
                    s_ += CONSONANT.get((mel - m) % 12, -0.8)
                if m == prev and pos != 0:
                    s_ -= 0.6  # hareket olsun
                if abs(m - prev) > 7:
                    s_ -= 0.8  # büyük sıçrama yapma
                s_ += rng.normal(0, 0.05)
                if s_ > best_s:
                    best, best_s = m, s_
            ev.append({"t": t0 + rng.normal(0, 0.004), "dur": max(0.08, (t1 - t0) * 0.92),
                       "midi": best, "vel": vel(0.95 if pos == 0 else 0.75, level)})
            prev = best
    return ev


def _drone_runs(bars, levels, min_level, clock, notes, base_vel):
    """Seviyesi yeterli ardışık ölçüleri tek uzun "dem" sesi olarak birleştirir."""
    ev = []
    run_start, run_levels = None, []
    for k, (bar, level) in enumerate(list(zip(bars, levels)) + [(None, -1)]):
        active = bar is not None and level >= min_level
        if active and run_start is None:
            run_start, run_levels = k, []
        if active:
            run_levels.append(level)
        if not active and run_start is not None:
            t0 = clock.time(bars[run_start][0])
            t1 = clock.time(bars[k - 1][-1] + 1)
            v = base_vel + 0.12 * float(np.mean(run_levels))
            for m in notes:
                ev.append({"t": t0, "dur": t1 - t0, "midi": m, "vel": v})
            run_start = None
    return ev


def _fold(midi, prev, lo, hi):
    """Perde sınıfını korur, çalgı aralığında önceki notaya en yakın oktavı seçer."""
    cands = [m for m in range(lo, hi + 1) if m % 12 == midi % 12]
    return min(cands, key=lambda m: abs(m - prev))


def _in_spans(t, spans):
    for a, b in spans:
        if a <= t < b:
            return (a, b)
    return None


# cura mızrap kalıbı: (vuruş içi konum, tel: 0 karar / 1 beşli / 2 tiz karar, hız)
CURA_PATTERNS = {
    2: {"sade": [(0.0, 0, 0.8), (1.0, 1, 0.6)],
        "dengeli": [(0.0, 0, 0.85), (0.5, 2, 0.45), (1.0, 1, 0.7), (1.5, 2, 0.45)],
        "zengin": [(0.0, 0, 0.9), (0.5, 2, 0.5), (1.0, 1, 0.75), (1.5, 2, 0.5)]},
    3: {"sade": [(0.0, 0, 0.8)],
        "dengeli": [(0.0, 0, 0.85), (1.0, 1, 0.6), (2.0, 1, 0.6)],
        "zengin": [(0.0, 0, 0.9), (1.0, 1, 0.65), (1.5, 2, 0.45), (2.0, 1, 0.65), (2.5, 2, 0.45)]},
    4: {"sade": [(0.0, 0, 0.8), (2.0, 1, 0.6)],
        "dengeli": [(0.0, 0, 0.85), (1.0, 2, 0.5), (2.0, 1, 0.7), (3.0, 2, 0.5)],
        "zengin": [(0.0, 0, 0.9), (0.5, 2, 0.45), (1.0, 2, 0.55), (2.0, 1, 0.75), (2.5, 2, 0.45),
                   (3.0, 2, 0.55)]},
}


def _cura_part(tonic, bars, levels, active, bpb, density, clock, rng):
    """Cura ikinci saz gibi çalar: karar ve beşli üzerinde usule göre ritmik mızrap (dem).
    Ezgiyi takip etmez; böylece analiz hatalarından etkilenmez ve hep makama uygundur."""
    root = 64 + (tonic - 64) % 12  # Mi4 .. Re#5 arası
    strings = [root, root + 7 if root + 7 <= 79 else root - 5, root + 12]
    ev = []
    for bar, level, on in zip(bars, levels, active):
        if not on:
            continue
        style = LEVEL_PATTERN[density][level]
        pat = CURA_PATTERNS[bpb][style]
        b0 = bar[0]
        for i, (pos, string, v) in enumerate(pat):
            t0 = clock.time(b0 + pos)
            nxt = pat[i + 1][0] if i + 1 < len(pat) else bpb
            t1 = clock.time(b0 + nxt)
            ev.append({"t": t0 + rng.normal(0, 0.005), "dur": max(0.08, t1 - t0),
                       "midi": strings[string], "vel": v * rng.uniform(0.9, 1.05)})
    return ev


def _smooth_melody(melody, min_dur):
    """Çok kısa notaları (süsleme / geçiş / analiz hatası) bir önceki notaya katar."""
    out = []
    for t, d, m in melody:
        if out and (d < min_dur or (m == out[-1][2] and t - (out[-1][0] + out[-1][1]) < 0.06)) \
                and t - (out[-1][0] + out[-1][1]) < 0.12:
            out[-1][1] = t + d - out[-1][0]
            continue
        if d < min_dur:
            continue
        out.append([t, d, m])
    return out


def _ney_octave(m, prev, lo=69, hi=88, center=78):
    """Neyi tiz ve sabit bir bölgede tutar: merkeze ve önceki notaya yakın oktav."""
    cands = [x for x in range(lo, hi + 1) if x % 12 == m % 12]
    return min(cands, key=lambda x: abs(x - prev) + 0.6 * abs(x - center))


def _ney_part(melody, spans, rng):
    """Ney ezgiyi uzun ve bağlı (legato) çalar; kısa süslemeleri atlar."""
    ev = []
    prev = 78
    for t, d, m in _smooth_melody(melody, 0.10):
        span = _in_spans(t, spans)
        if span is None:
            continue
        note = _ney_octave(m, prev)
        prev = note
        dur = min(d, span[1] - t)
        if ev:
            gap = t - (ev[-1]["t"] + ev[-1]["dur"])
            if ev[-1]["midi"] == note and gap < 0.06:
                ev[-1]["dur"] = t + dur - ev[-1]["t"]
                continue
            if 0 < gap < 0.6:
                ev[-1]["dur"] = t - ev[-1]["t"] + 0.03  # legato: boşluğu kapat, hafif üst üste bin
        ev.append({"t": t + 0.015, "dur": dur, "midi": note, "vel": 0.7 * rng.uniform(0.92, 1.04)})
    # cümle sonlarında sesi biraz uzat (nefes bitene kadar)
    for i, e in enumerate(ev):
        nxt = ev[i + 1]["t"] if i + 1 < len(ev) else e["t"] + e["dur"] + 0.6
        e["dur"] = max(e["dur"], min(nxt - e["t"] + 0.05, e["dur"] + 0.8))
    return ev


def arrange(an, settings):
    density = settings.get("density", "dengeli")
    instruments = settings.get("instruments", DEFAULT_INSTRUMENTS)
    seed = int(settings.get("seed", 1))
    rng = np.random.default_rng(seed)

    beats, strength = apply_tempo_factor(an["beats"], an["low_strength"], float(settings.get("tempo_factor", 1.0)))
    if len(beats) < 4:
        return {"events": {}, "bars": 0, "info": "Yeterli ritim bulunamadı"}

    bpb = int(settings.get("beats_per_bar") or an["beats_per_bar"])
    if bpb not in (2, 3, 4):
        bpb = 4
    phase = downbeat_phase(np.asarray(strength), bpb)
    tonic = settings.get("tonic_pc")
    tonic = an["tonic_pc"] if tonic is None else int(tonic)
    quality = an["quality"]

    # vuruş enerjisini (tempo değişse de) zamana göre eşle
    src_beats = np.asarray(an["beats"], dtype=float)
    energy = np.interp(beats, src_beats, np.asarray(an["beat_energy"], dtype=float))

    bars = []
    i = phase
    while i + bpb <= len(beats):
        bars.append(list(range(i, i + bpb)))
        i += bpb
    if not bars:
        return {"events": {}, "bars": 0, "info": "Ölçü bulunamadı"}
    levels = _bar_levels(energy, bars)
    clock = BeatClock(beats)

    def jitter(sec):
        return sec + rng.normal(0, 0.004)

    def vel(v, level):
        return float(np.clip(v * (0.8 + 0.12 * level) * rng.uniform(0.92, 1.08), 0.05, 1.0))

    events = {}

    scale = _scale_from_chroma(an.get("chroma", [1] * 12), tonic)
    roots = _bar_roots(bars, clock, an.get("melody") or [], tonic, scale)

    if "bas" in instruments:
        root = _bass_root(tonic)
        ev = _bass_line(bars, levels, roots, bpb, density, clock, an.get("melody") or [], tonic, scale, rng, vel)
        # bitiş: son ölçüden sonra uzun karar sesi
        end_t = clock.time(bars[-1][-1] + 1)
        ev.append({"t": end_t, "dur": 2.0 * clock.period, "midi": root, "vel": 0.85})
        events["bas"] = ev

    if "bendir" in instruments:
        events["bendir"] = _bendir_part(bars, levels, bpb, density, clock, rng)

    for kit in ("darbuka",):
        if kit not in instruments:
            continue
        ev = []
        for bar, level in zip(bars, levels):
            style = LEVEL_PATTERN[density][level]
            if kit == "darbuka" and "bendir" in instruments and style == "sade":
                continue  # ikisi birlikteyken sakin bölümlerde yalnız bendir
            pat = PERC_PATTERNS[bpb][style]
            b0 = bar[0]
            for pos, kind, v in pat:
                if kit == "bendir" and kind == "ka" and style != "zengin":
                    continue  # bendir daha sade: ara vuruşlar yalnız zengin yoğunlukta
                ev.append({"t": jitter(clock.time(b0 + pos)), "kind": kind, "vel": vel(v, level)})
        ev.append({"t": clock.time(bars[-1][-1] + 1), "kind": "dum", "vel": 1.0})
        events[kit] = ev

    min_level = {"sade": 2, "dengeli": 1, "zengin": 0}[density]

    if "org" in instruments:
        root = _pad_root(tonic)  # org dem'i orta bölgede (16' çekmece zaten pes katman ekler)
        notes = [root, root + 7, root + 12]
        if density == "zengin":
            notes.append(root + (3 if quality == "minor" else 4) + 12)
        events["org"] = _drone_runs(bars, levels, min_level, clock, notes, 0.5)

    if "yaylilar" in instruments:
        root = _pad_root(tonic) + (12 if "org" in instruments else 0)  # org varsa üstte
        notes = [root, root + 7, root + 12]
        if density == "zengin":
            notes.append(root + (3 if quality == "minor" else 4) + 12)
        # org da varsa yaylılar yalnız yoğun bölümlere girer
        lvl = min(2, min_level + 1) if "org" in instruments else min_level
        events["yaylilar"] = _drone_runs(bars, levels, lvl, clock, notes, 0.55)

    melody = an.get("melody") or []
    has_cura, has_ney = "cura" in instruments, "ney" in instruments
    if has_cura or has_ney:
        cura_on, ney_spans = [], []
        for bar, level in zip(bars, levels):
            eligible = level >= min_level
            if has_cura and has_ney:
                # yoğun bölümlerde cura ritim tutar, sakin bölümlerde ney ezgiye eşlik eder
                # cura ritim tuttuğu için ney ile çakışmaz: cura yoğun bölümlerde, ney her yerde
                cura_on.append(eligible and level == 2)
                if eligible:
                    ney_spans.append((clock.time(bar[0]), clock.time(bar[-1] + 1)))
            else:
                cura_on.append(eligible)
                if eligible:
                    ney_spans.append((clock.time(bar[0]), clock.time(bar[-1] + 1)))
        if has_cura:
            events["cura"] = _cura_part(tonic, bars, levels, cura_on, bpb, density, clock, rng)
        if has_ney and melody:
            events["ney"] = _ney_part(melody, ney_spans, rng)

    return {
        "events": events,
        "bars": len(bars),
        "beats_per_bar": bpb,
        "bpm": round(60.0 / clock.period, 1),
        "tonic_pc": tonic,
        "levels": [int(v) for v in levels],
    }
