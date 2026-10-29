"""16-bit PCM WAV okuma/yazma (yalnız standart kütüphane + NumPy)."""

import wave

import numpy as np


def read_wav(path):
    """WAV dosyasını okur. (örnekler[float32, şekil (n, kanal)], örnekleme hızı) döner."""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("Yalnız 16-bit WAV destekleniyor (gelen: %d bit)" % (width * 8))
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return data.reshape(-1, ch), sr


def write_wav(path, samples, sr):
    """float dizi ([-1, 1]) -> 16-bit WAV. samples: (n,) mono veya (n, kanal)."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    ch = x.shape[1]
    pcm = np.clip(x, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


def to_mono(x):
    return x.mean(axis=1) if x.ndim == 2 else x
