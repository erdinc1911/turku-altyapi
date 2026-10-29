"""SF2 / SF3 (Ogg sıkıştırmalı) ses bankası okuyucu. Yalnız derleme aracı olarak masaüstünde kullanılır.

SF3 örneklerini çözmek için sistemde `ffmpeg` gerekir.
"""

import struct
import subprocess

import numpy as np

GEN_KEYRANGE, GEN_VELRANGE, GEN_INSTRUMENT, GEN_SAMPLEID = 43, 44, 41, 53
GEN_COARSE, GEN_FINE, GEN_SAMPLEMODES, GEN_ROOTKEY = 51, 52, 54, 58
GEN_ATTENUATION = 48


def _chunks(data, start, end):
    pos = start
    while pos + 8 <= end:
        cid = data[pos:pos + 4].decode("latin1")
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        yield cid, pos + 8, size
        pos += 8 + size + (size & 1)


class SoundFont:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = data = f.read()
        assert data[:4] == b"RIFF" and data[8:12] == b"sfbk"
        self.smpl = None
        pdta = {}
        for cid, off, size in _chunks(data, 12, len(data)):
            if cid != "LIST":
                continue
            kind = data[off:off + 4].decode()
            for sub, soff, ssize in _chunks(data, off + 4, off + size):
                if kind == "sdta" and sub == "smpl":
                    self.smpl = (soff, ssize)
                elif kind == "pdta":
                    pdta[sub] = data[soff:soff + ssize]
        self.presets = self._records(pdta["phdr"], 38, "<20sHHHIII")
        self.pbag = self._records(pdta["pbag"], 4, "<HH")
        self.pgen = self._records(pdta["pgen"], 4, "<HH")
        self.inst = self._records(pdta["inst"], 22, "<20sH")
        self.ibag = self._records(pdta["ibag"], 4, "<HH")
        self.igen = self._records(pdta["igen"], 4, "<HH")
        self.shdr = self._records(pdta["shdr"], 46, "<20sIIIIIBbHH")

    @staticmethod
    def _records(raw, size, fmt):
        return [struct.unpack(fmt, raw[i:i + size]) for i in range(0, len(raw) - size + 1, size)]

    @staticmethod
    def _name(b):
        return b.split(b"\0")[0].decode("latin1").strip()

    def list_presets(self):
        return [(p[2], p[1], self._name(p[0])) for p in self.presets[:-1]]

    def _gens(self, bags, gens, b0, b1):
        zones = []
        for b in range(b0, b1):
            g0, g1 = bags[b][0], bags[b + 1][0]
            z = {}
            for op, amt in gens[g0:g1]:
                z[op] = amt
            zones.append(z)
        return zones

    def preset_zones(self, bank, program):
        """Önayardaki tüm (çalgı bölgesi) kayıtlarını, örnek bilgisiyle birlikte döner."""
        idx = [i for i, p in enumerate(self.presets[:-1]) if p[2] == bank and p[1] == program]
        if not idx:
            raise KeyError("önayar yok: %d/%d" % (bank, program))
        i = idx[0]
        pz = self._gens(self.pbag, self.pgen, self.presets[i][3], self.presets[i + 1][3])
        out = []
        for z in pz:
            if GEN_INSTRUMENT not in z:
                continue
            pkr = z.get(GEN_KEYRANGE, 0x7F00)
            inst_i = z[GEN_INSTRUMENT]
            iz = self._gens(self.ibag, self.igen, self.inst[inst_i][1], self.inst[inst_i + 1][1])
            global_z = iz[0] if iz and GEN_SAMPLEID not in iz[0] else {}
            for zz in iz:
                if GEN_SAMPLEID not in zz:
                    continue
                g = dict(global_z)
                g.update(zz)
                kr = g.get(GEN_KEYRANGE, 0x7F00)
                lo, hi = kr & 0xFF, kr >> 8
                plo, phi = pkr & 0xFF, pkr >> 8
                lo, hi = max(lo, plo), min(hi, phi)
                if lo > hi:
                    continue
                vr = g.get(GEN_VELRANGE, 0x7F00)
                s = self.shdr[g[GEN_SAMPLEID]]
                root = g.get(GEN_ROOTKEY, 0xFFFF)
                if root > 127:
                    root = s[6]
                coarse = struct.unpack("<h", struct.pack("<H", g.get(GEN_COARSE, 0)))[0]
                fine = struct.unpack("<h", struct.pack("<H", g.get(GEN_FINE, 0)))[0]
                out.append({
                    "inst": self._name(self.inst[inst_i][0]),
                    "sample": self._name(s[0]),
                    "sample_index": g[GEN_SAMPLEID],
                    "key_lo": lo, "key_hi": hi,
                    "vel_lo": vr & 0xFF, "vel_hi": vr >> 8,
                    "root": root,
                    # kök perdeden sent cinsinden sapma (örnek düzeltmesi + ayar)
                    "tune_cents": s[7] + coarse * 100 + fine,
                    "loop_mode": g.get(GEN_SAMPLEMODES, 0) & 3,
                    "attenuation_cb": g.get(GEN_ATTENUATION, 0),
                })
        return out

    def sample_audio(self, index):
        """Örneği float32 mono olarak çözer. (ses, örnekleme hızı, döngü başı, döngü sonu) döner."""
        name, start, end, sloop, eloop, sr, _, _, _, stype = self.shdr[index]
        off = self.smpl[0]
        if stype & 0x10:  # SF3: Ogg Vorbis; start/end bayt, döngü örnek başından itibaren
            blob = self.data[off + start: off + end]
            proc = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "pipe:1"],
                input=blob, capture_output=True, check=True)
            audio = np.frombuffer(proc.stdout, dtype="<f4").copy()
            return audio, sr, sloop, eloop
        raw = self.data[off + 2 * start: off + 2 * end]
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        return audio, sr, sloop - start, eloop - start
