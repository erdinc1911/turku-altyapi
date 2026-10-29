"""Müzikal analiz: tempo, vuruşlar, ölçü başı, akort sapması, karar sesi, enerji.

Yöntemler bilinçli olarak sade ve yalnız NumPy ile yazıldı:
- Başlangıç (onset) zarfı: log-genlik spektral akı.
- Tempo: başlangıç zarfının otokorelasyonu + geniş bir tempo önsel dağılımı.
- Vuruş takibi: dinamik programlama (Ellis, 2007).
- Akort: spektral tepe frekanslarının 440 Hz ızgarasından dairesel ortalama sapması.
- Karar sesi: kroma histogramı + parçanın sonundaki seslere ve bas bölgesine ek ağırlık.
"""

import numpy as np

ANALYSIS_SR = 22050
N_FFT = 2048
HOP = 512

NOTE_NAMES = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]


def _noop(pct, msg):
    pass


# ---------------------------------------------------------------- ön işleme

def _lowpass_kernel(cutoff, taps=63):
    n = np.arange(taps) - (taps - 1) / 2.0
    h = np.sinc(2 * cutoff * n) * np.hanning(taps)
    return (h / h.sum()).astype(np.float32)


def resample(x, sr, target):
    """Basit yeniden örnekleme: alçak geçiren süzgeç + doğrusal ara değer."""
    if sr == target:
        return x.astype(np.float32)
    if target < sr:
        x = np.convolve(x, _lowpass_kernel(0.5 * target / sr * 0.9), mode="same")
    n_out = int(len(x) * target / sr)
    t = np.arange(n_out, dtype=np.float64) * (sr / target)
    return np.interp(t, np.arange(len(x)), x).astype(np.float32)


def stft_mag(y, progress=_noop, p0=0, p1=0):
    """Genlik spektrogramı (çerçeve x frekans), parça parça hesaplanır (bellek dostu)."""
    pad = N_FFT // 2
    y = np.pad(y, (pad, pad))
    n_frames = 1 + (len(y) - N_FFT) // HOP
    win = np.hanning(N_FFT).astype(np.float32)
    mag = np.empty((n_frames, N_FFT // 2 + 1), dtype=np.float32)
    chunk = 512
    for start in range(0, n_frames, chunk):
        stop = min(n_frames, start + chunk)
        idx = np.arange(start, stop)[:, None] * HOP + np.arange(N_FFT)[None, :]
        frames = y[idx] * win
        mag[start:stop] = np.abs(np.fft.rfft(frames, axis=1))
        if p1 > p0:
            progress(int(p0 + (p1 - p0) * stop / n_frames), "Spektrum hesaplanıyor")
    return mag


def _bin_freqs():
    return np.arange(N_FFT // 2 + 1) * ANALYSIS_SR / N_FFT


def onset_envelope(mag, fmin=30.0, fmax=8000.0):
    f = _bin_freqs()
    band = (f >= fmin) & (f <= fmax)
    logm = np.log1p(100.0 * mag[:, band])
    flux = np.maximum(0.0, np.diff(logm, axis=0)).sum(axis=1)
    flux = np.concatenate([[0.0], flux]).astype(np.float64)
    # yerel ortalamayı çıkar (yaklaşık 0,5 sn pencere)
    w = 21
    local = np.convolve(flux, np.ones(w) / w, mode="same")
    env = np.maximum(0.0, flux - local)
    sd = env.std()
    return env / sd if sd > 0 else env


# ---------------------------------------------------------------- tempo ve vuruş

def estimate_tempo(env, fr, bpm_min=50.0, bpm_max=220.0, prior_center=110.0, prior_octaves=1.0):
    n = len(env)
    e = env - env.mean()
    spec = np.fft.rfft(e, 2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    if ac[0] <= 0:
        return prior_center
    ac = ac / ac[0]

    def ac_at(lag):
        return np.interp(lag, np.arange(n), ac, right=0.0)

    bpms = np.linspace(bpm_min, bpm_max, 600)
    lags = 60.0 * fr / bpms
    score = ac_at(lags) + 0.5 * ac_at(2 * lags) + 0.25 * ac_at(lags / 2)
    prior = np.exp(-0.5 * (np.log2(bpms / prior_center) / prior_octaves) ** 2)
    total = score * prior
    return float(bpms[int(np.argmax(total))])


def track_beats(env, fr, bpm, tightness=100.0):
    """Ellis (2007) dinamik programlama vuruş takibi. Çerçeve indisleri döner."""
    period = 60.0 * fr / bpm
    # yerel skor: vuruş periyoduna göre yumuşatılmış zarf
    sigma = max(1.0, period / 32.0)
    k = np.arange(-int(4 * sigma), int(4 * sigma) + 1)
    g = np.exp(-0.5 * (k / sigma) ** 2)
    local = np.convolve(env, g / g.sum(), mode="same")

    n = len(local)
    offsets = np.arange(-int(round(2 * period)), -int(round(period / 2)) + 1)
    txwt = -tightness * (np.log(-offsets / period)) ** 2
    cum = np.zeros(n)
    back = -np.ones(n, dtype=np.int64)
    for t in range(n):
        prev = t + offsets
        valid = prev >= 0
        if np.any(valid):
            cand = cum[prev[valid]] + txwt[valid]
            j = int(np.argmax(cand))
            best = cand[j]
            if best > 0:
                cum[t] = local[t] + best
                back[t] = prev[valid][j]
                continue
        cum[t] = local[t]

    # son vuruş: son periyot içindeki en yüksek kümülatif skor
    tail = max(1, int(round(period)))
    end = n - tail + int(np.argmax(cum[n - tail:]))
    beats = [end]
    while back[beats[-1]] >= 0:
        beats.append(int(back[beats[-1]]))
    return np.array(beats[::-1], dtype=np.int64)


def trim_beats(beat_frames, frame_energy, rel_db=-30.0):
    """Sessiz baş ve son kısımdaki vuruşları atar."""
    if len(beat_frames) == 0:
        return beat_frames
    e = frame_energy[np.clip(beat_frames, 0, len(frame_energy) - 1)]
    ref = np.percentile(frame_energy, 95) + 1e-12
    db = 20 * np.log10(e / ref + 1e-12)
    loud = np.where(db > rel_db)[0]
    if len(loud) == 0:
        return beat_frames
    return beat_frames[loud[0]: loud[-1] + 1]


def estimate_meter(beat_strength):
    """Ölçüdeki vuruş sayısı tahmini (3 veya 4; 2/4 eserler 4 ile uyumlu çalışır)."""
    s = beat_strength - beat_strength.mean()
    if len(s) < 16 or s.std() == 0:
        return 4

    def corr(lag):
        return float(np.dot(s[:-lag], s[lag:]) / (len(s) - lag))

    c3, c2, c4 = corr(3), corr(2), corr(4)
    return 3 if c3 > 1.15 * max(c2, c4, 1e-9) else 4


def downbeat_phase(low_strength, bpb):
    scores = [low_strength[k::bpb].mean() if len(low_strength[k::bpb]) else 0.0 for k in range(bpb)]
    return int(np.argmax(scores))


# ---------------------------------------------------------------- akort ve karar sesi

def estimate_tuning(mag, step=4):
    """440 Hz eşit akort ızgarasından ortalama sapma (sent, -50..+50)."""
    f = _bin_freqs()
    lo, hi = np.searchsorted(f, 100.0), np.searchsorted(f, 2000.0)
    m = mag[::step, lo - 1: hi + 1].astype(np.float64)
    a, b, c = m[:, :-2], m[:, 1:-1], m[:, 2:]
    frame_max = b.max(axis=1, keepdims=True) + 1e-12
    peaks = (b > a) & (b >= c) & (b > 0.1 * frame_max)
    if not np.any(peaks):
        return 0.0
    denom = a - 2 * b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(denom != 0, 0.5 * (a - c) / denom, 0.0)
    bins = np.arange(lo, hi)[None, :] + delta
    freq = bins * ANALYSIS_SR / N_FFT
    sel = peaks & (freq > 0)
    cents = 1200 * np.log2(freq[sel] / 440.0)
    dev = ((cents + 50.0) % 100.0) - 50.0
    w = b[sel]
    vec = np.sum(w * np.exp(1j * 2 * np.pi * dev / 100.0))
    return float(np.angle(vec) / (2 * np.pi) * 100.0)


def _chroma_matrix(tuning_cents, fmin, fmax):
    f = _bin_freqs()
    W = np.zeros((len(f), 12), dtype=np.float32)
    ok = (f >= fmin) & (f <= fmax)
    midi = 69 + 12 * np.log2(np.where(ok, f, 440.0) / 440.0) - tuning_cents / 100.0
    pc = np.mod(np.round(midi).astype(int), 12)
    W[np.where(ok)[0], pc[ok]] = 1.0
    return W


def estimate_tonic(mag, frame_energy, tuning_cents):
    def hist(m, W):
        ch = m @ W
        ch = ch / (ch.sum(axis=1, keepdims=True) + 1e-9)
        return (ch * frame_energy[: len(ch), None]).sum(axis=0)

    W_full = _chroma_matrix(tuning_cents, 60.0, 2000.0)
    W_low = _chroma_matrix(tuning_cents, 50.0, 300.0)
    active = np.where(frame_energy > np.percentile(frame_energy, 30))[0]
    if len(active) < 10:
        active = np.arange(len(frame_energy))
    end_frames = active[int(len(active) * 0.88):]

    h_full = hist(mag[active], W_full)
    h_end = hist(mag[end_frames], W_full)
    h_low = hist(mag[active], W_low)

    def norm(h):
        s = h.sum()
        return h / s if s > 0 else h

    score = 0.5 * norm(h_full) + 0.3 * norm(h_end) + 0.2 * norm(h_low)
    tonic = int(np.argmax(score))
    hf = norm(h_full)
    quality = "minor" if hf[(tonic + 3) % 12] >= hf[(tonic + 4) % 12] else "major"
    return tonic, quality, [round(float(v), 4) for v in hf]


# ---------------------------------------------------------------- melodi

MEL_FFT = 4096


def extract_melody(y, tuning_cents, progress=_noop, p0=0, p1=0):
    """Baskın ezgiyi (vokal + saz birlikte) nota dizisi olarak çıkarır.

    Harmonik toplam (salience) yöntemi: her aday temel frekans için ilk 8 harmoniğin
    ağırlıklı genliği toplanır, en yüksek aday seçilir. Oktav hataları sonradan önemsizdir;
    çalgılar ezgiyi kendi oktav aralıklarına yerleştirir.
    Notalar ölçülen akort ızgarasına (12 eşit ses) yuvarlanır; koma perdeleri en yakın
    yarım sese düşer (ileride makam desteğiyle düzeltilecek).
    """
    pad = MEL_FFT // 2
    yp = np.pad(y, (pad, pad))
    n_frames = 1 + (len(yp) - MEL_FFT) // HOP
    win = np.hanning(MEL_FFT).astype(np.float32)
    binw = ANALYSIS_SR / MEL_FFT

    cand = np.arange(45.0, 86.0, 0.1)  # La2 .. Re6 aralığı, 10 sent adım
    f0 = 440.0 * 2 ** ((cand - 69) / 12.0 + tuning_cents / 1200.0)
    H = 8
    weights = 0.8 ** np.arange(H)
    nbins = MEL_FFT // 2 + 1
    pos = f0[:, None] * np.arange(1, H + 1)[None, :] / binw
    valid = pos < nbins - 2
    pos = np.where(valid, pos, 0)
    i0 = pos.astype(np.int64)
    fr = (pos - i0).astype(np.float32)
    w = (weights[None, :] * valid).astype(np.float32)

    best = np.zeros(n_frames, dtype=np.int64)
    best_sal = np.zeros(n_frames, dtype=np.float32)
    energy = np.zeros(n_frames, dtype=np.float32)
    chunk = 128
    for start in range(0, n_frames, chunk):
        stop = min(n_frames, start + chunk)
        idx = np.arange(start, stop)[:, None] * HOP + np.arange(MEL_FFT)[None, :]
        mag = np.abs(np.fft.rfft(yp[idx] * win, axis=1)).astype(np.float32)
        mag = np.sqrt(mag)  # sıkıştırma: güçlü tek harmoniğin baskınlığını azaltır
        sal = np.zeros((stop - start, len(cand)), dtype=np.float32)
        for h in range(H):
            a = mag[:, i0[:, h]]
            b = mag[:, i0[:, h] + 1]
            sal += w[:, h] * (a + (b - a) * fr[:, h])
        best[start:stop] = np.argmax(sal, axis=1)
        best_sal[start:stop] = sal[np.arange(stop - start), best[start:stop]]
        energy[start:stop] = mag.sum(axis=1)
        if p1 > p0:
            progress(int(p0 + (p1 - p0) * stop / n_frames), "Ezgi çıkarılıyor")

    voiced = (best_sal > 0.35 * np.percentile(best_sal, 90)) & (energy > np.percentile(energy, 25))
    midi = np.round(cand[best]).astype(np.int64)
    # 5 karelik medyan süzgeç (yalnız sesli kareler)
    sm = midi.copy()
    for k in range(2, len(midi) - 2):
        sm[k] = int(np.median(midi[k - 2:k + 3]))
    sm[~voiced] = -1

    notes = []
    k = 0
    frame_t = HOP / ANALYSIS_SR
    while k < len(sm):
        if sm[k] < 0:
            k += 1
            continue
        j = k
        while j + 1 < len(sm) and (sm[j + 1] == sm[k] or (sm[j + 1] < 0 and j + 2 < len(sm) and sm[j + 2] == sm[k])):
            j += 1
        if j - k + 1 >= 3:
            notes.append([round(k * frame_t, 3), round((j - k + 1) * frame_t, 3), int(sm[k])])
        k = j + 1
    return notes


# ---------------------------------------------------------------- ana giriş

def analyze(y_mono, sr, progress=_noop):
    """Tam analiz. Sonuç JSON'a yazılabilir bir sözlüktür (zamanlar saniye)."""
    progress(2, "Ses hazırlanıyor")
    y = resample(y_mono, sr, ANALYSIS_SR)
    duration = len(y_mono) / float(sr)

    mag = stft_mag(y, progress, 5, 30)
    fr = ANALYSIS_SR / HOP
    frame_energy = mag.sum(axis=1).astype(np.float64)

    progress(32, "Ritim aranıyor")
    env = onset_envelope(mag)
    low_env = onset_envelope(mag, 30.0, 300.0)
    bpm = estimate_tempo(env, fr)

    progress(40, "Vuruşlar takip ediliyor")
    beats = track_beats(env, fr, bpm)
    beats = trim_beats(beats, frame_energy)

    if len(beats) >= 2:
        bpm = float(60.0 * fr / np.median(np.diff(beats)))

    strength = env[beats] if len(beats) else np.zeros(0)
    low_strength = low_env[beats] if len(beats) else np.zeros(0)
    bpb = estimate_meter(strength) if len(beats) else 4

    progress(50, "Akort ve karar sesi bulunuyor")
    tuning = estimate_tuning(mag)
    tonic, quality, chroma = estimate_tonic(mag, frame_energy, tuning)
    del mag
    melody = extract_melody(y, tuning, progress, 55, 95)

    # her vuruşun enerjisi (dinamik için)
    beat_energy = []
    for i, b in enumerate(beats):
        e_end = beats[i + 1] if i + 1 < len(beats) else min(len(frame_energy), b + 20)
        seg = frame_energy[b: max(b + 1, e_end)]
        beat_energy.append(float(seg.mean()))

    # Spektral akı, Hann penceresinin dik kenarında tepe yapar: başlangıçlar yaklaşık
    # çeyrek pencere erken görünür. Bunu telafi et.
    latency = N_FFT / (4.0 * ANALYSIS_SR)
    times = (beats * HOP / ANALYSIS_SR + latency).astype(float)
    return {
        "duration": duration,
        "sample_rate": sr,
        "bpm": round(bpm, 2),
        "beats": [round(float(t), 4) for t in times],
        "beat_strength": [round(float(v), 4) for v in strength],
        "low_strength": [round(float(v), 4) for v in low_strength],
        "beat_energy": beat_energy,
        "beats_per_bar": int(bpb),
        "downbeat_phase": downbeat_phase(low_strength, bpb) if len(beats) else 0,
        "tuning_cents": round(tuning, 1),
        "tonic_pc": tonic,
        "tonic_name": NOTE_NAMES[tonic],
        "quality": quality,
        "chroma": chroma,
        "melody": melody,
    }
