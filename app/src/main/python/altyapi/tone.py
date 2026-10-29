"""Referans kayda göre ses rengi (frekans dengesi) eşleme.

Referans kaydın kendisi saklanmaz; yalnız 8 frekans bandındaki enerji dağılımı ölçülür.
Altyapı stemlerine, toplam miksin dağılımını referansa yaklaştıran yumuşak bir
eşitleyici (EQ) uygulanır. Kaynak kayda (kullanıcının sesi ve sazı) dokunulmaz.
"""

import json

import numpy as np

BANDS = [30, 90, 180, 350, 700, 1400, 2800, 5600, 16000]  # 8 bant sınırı (Hz)
BAND_NAMES = ["çok pes", "pes", "alt orta-pes", "orta-pes", "orta", "üst orta", "tiz", "çok tiz"]
MAX_CUT_DB, MAX_BOOST_DB = -9.0, 6.0


def band_energy(x, sr):
    """Bant başına toplam enerji (mutlak, karşılaştırma için)."""
    if x.ndim == 2:
        x = x.mean(axis=1)
    n_fft = 4096
    hop = 2048
    win = np.hanning(n_fft).astype(np.float32)
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    masks = [(f >= BANDS[i]) & (f < BANDS[i + 1]) for i in range(len(BANDS) - 1)]
    e = np.zeros(len(masks))
    n_frames = max(0, 1 + (len(x) - n_fft) // hop)
    chunk = 256
    for s0 in range(0, n_frames, chunk):
        s1 = min(n_frames, s0 + chunk)
        idx = np.arange(s0, s1)[:, None] * hop + np.arange(n_fft)[None, :]
        P = np.abs(np.fft.rfft(x[idx] * win, axis=1)) ** 2
        for i, m in enumerate(masks):
            e[i] += P[:, m].sum()
    return e


def profile_db(energy):
    """Enerji dağılımını, toplamı 1 olan paylara ve dB'ye çevirir."""
    e = np.asarray(energy, dtype=float)
    share = e / max(e.sum(), 1e-12)
    return 10 * np.log10(share + 1e-9)


def reference_profile(wav_path, out_json):
    from . import wavio
    x, sr = wavio.read_wav(wav_path)
    prof = profile_db(band_energy(x, sr))
    data = {"bands_hz": BANDS, "profile_db": [round(float(v), 2) for v in prof]}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return json.dumps(data)


def eq_gains(ref_db, source_energy, stems_energy):
    """Stemlere uygulanacak bant kazançları (dB).

    Hedef: (kaynak + EQ'lu stemler) toplamının bant dağılımı referansa yaklaşsın.
    Kaynak sabit olduğu için yalnız stemlerin katkısı ayarlanır.
    """
    src = np.asarray(source_energy, dtype=float)
    stm = np.asarray(stems_energy, dtype=float)
    total = src + stm
    target_share = 10 ** (np.asarray(ref_db) / 10.0)
    target = target_share * total.sum()  # aynı toplam enerji, referans dağılımı
    gains = np.zeros(len(stm))
    for i in range(len(stm)):
        if stm[i] <= total[i] * 0.02:  # stemlerin bu bantta neredeyse hiç payı yok
            continue
        want = (target[i] - src[i]) / stm[i]
        want = max(want, 10 ** (MAX_CUT_DB / 10.0))
        gains[i] = 10 * np.log10(want)
    gains = np.clip(gains, MAX_CUT_DB, MAX_BOOST_DB)
    # komşu bantlarla yumuşat (keskin eşitleyici doğal duyulmaz)
    sm = gains.copy()
    for i in range(len(gains)):
        sm[i] = 0.5 * gains[i] + 0.25 * gains[max(0, i - 1)] + 0.25 * gains[min(len(gains) - 1, i + 1)]
    return sm


def apply_eq(x, sr, gains_db):
    """Bant kazançlarını, log-frekansta yumuşak geçişli doğrusal fazlı bir süzgeçle uygular."""
    if np.allclose(gains_db, 0):
        return x
    taps = 2049
    nfft_design = 8192
    f = np.fft.rfftfreq(nfft_design, 1.0 / sr)
    centers = np.sqrt(np.array(BANDS[:-1]) * np.array(BANDS[1:]))
    g = np.interp(np.log10(np.maximum(f, 20.0)), np.log10(centers), gains_db)
    H = 10 ** (g / 20.0)
    h = np.fft.irfft(H, nfft_design)
    h = np.roll(h, taps // 2)[:taps] * np.hanning(taps)
    block = 1 << 16
    nfft = 1 << int(np.ceil(np.log2(block + taps - 1)))
    Hf = np.fft.rfft(h, nfft)
    out = np.zeros(len(x) + taps, dtype=np.float32)
    for s in range(0, len(x), block):
        seg = x[s:s + block]
        y = np.fft.irfft(np.fft.rfft(seg, nfft) * Hf, nfft)[: len(seg) + taps - 1]
        out[s:s + len(y)] += y.astype(np.float32)
    d = taps // 2  # doğrusal faz gecikmesini geri al
    return out[d:d + len(x)]


def describe(gains_db):
    """Kullanıcıya sade bir açıklama."""
    lo = float(np.mean(gains_db[:3]))
    hi = float(np.mean(gains_db[5:]))
    parts = []
    if hi < -2:
        parts.append("tizler kısıldı (daha sıcak)")
    elif hi > 2:
        parts.append("tizler açıldı (daha parlak)")
    if lo > 2:
        parts.append("pesler güçlendi")
    elif lo < -2:
        parts.append("pesler azaltıldı")
    return ", ".join(parts) if parts else "küçük ayarlar yapıldı"
