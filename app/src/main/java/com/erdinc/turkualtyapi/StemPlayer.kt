package com.erdinc.turkualtyapi

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import java.io.OutputStream
import kotlin.math.abs
import kotlin.math.max

/**
 * Kaynak kayıt ve üretilen stemleri eşzamanlı çalan basit mikser.
 * Her iz için ses seviyesi ve susturma anlık değiştirilebilir.
 */
class StemPlayer(val tracks: List<Track>) {

    class Track(val key: String, val name: String, val data: WavFile.Data, gain: Float) {
        @Volatile var gain: Float = gain
        @Volatile var muted: Boolean = false
    }

    val sampleRate: Int = tracks.first().data.sampleRate
    val totalFrames: Int = tracks.maxOf { it.data.frames }

    @Volatile var position: Int = 0
        private set
    @Volatile var isPlaying = false
        private set

    private var thread: Thread? = null
    var onPosition: ((Int) -> Unit)? = null
    var onFinished: (() -> Unit)? = null

    fun seekFraction(f: Float) {
        position = (f.coerceIn(0f, 1f) * totalFrames).toInt()
    }

    fun play() {
        if (isPlaying) return
        if (position >= totalFrames) position = 0
        isPlaying = true
        thread = Thread({ loop() }, "stem-player").also { it.start() }
    }

    fun pause() {
        isPlaying = false
        thread?.join(500)
        thread = null
    }

    fun release() = pause()

    private fun loop() {
        val minBuf = AudioTrack.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_OUT_STEREO, AudioFormat.ENCODING_PCM_16BIT
        )
        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC).build()
            )
            .setAudioFormat(
                AudioFormat.Builder().setSampleRate(sampleRate)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO).build()
            )
            .setBufferSizeInBytes(max(minBuf, 8192) * 2)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
        val block = 2048
        val out = ShortArray(block * 2)
        track.play()
        try {
            var tick = 0
            while (isPlaying && position < totalFrames) {
                val n = minOf(block, totalFrames - position)
                mix(position, n, out, 1f)
                track.write(out, 0, n * 2)
                position += n
                if (++tick % 4 == 0) onPosition?.invoke(position)
            }
        } finally {
            track.stop(); track.release()
        }
        if (position >= totalFrames) {
            isPlaying = false
            onPosition?.invoke(position)
            onFinished?.invoke()
        }
    }

    /** start karesinden itibaren n kareyi stereo olarak karıştırır. */
    private fun mix(start: Int, n: Int, out: ShortArray, scale: Float) {
        val l = FloatArray(n)
        val r = FloatArray(n)
        for (t in tracks) {
            if (t.muted || t.gain <= 0f) continue
            val g = t.gain * scale
            val d = t.data
            val ch = d.channels
            val s = d.samples
            for (i in 0 until n) {
                val f = start + i
                if (f >= d.frames) break
                if (ch == 1) {
                    val v = s[f] * g
                    l[i] += v; r[i] += v
                } else {
                    l[i] += s[f * ch] * g
                    r[i] += s[f * ch + 1] * g
                }
            }
        }
        for (i in 0 until n) {
            out[2 * i] = l[i].coerceIn(-32768f, 32767f).toInt().toShort()
            out[2 * i + 1] = r[i].coerceIn(-32768f, 32767f).toInt().toShort()
        }
    }

    /** Tüm parçayı mevcut ayarlarla karıştırıp WAV olarak yazar (taşma varsa ses düşürülür). */
    fun exportWav(stream: OutputStream) {
        val block = 8192
        val tmp = ShortArray(block * 2)
        // 1. geçiş: tepe değeri bul (kırpmadan önce)
        var peak = 1f
        val probe = FloatArray(block * 2)
        var p = 0
        while (p < totalFrames) {
            val n = minOf(block, totalFrames - p)
            peakOf(p, n, probe).let { peak = max(peak, it) }
            p += n
        }
        val scale = if (peak > 32000f) 32000f / peak else 1f
        // 2. geçiş: blok blok yaz (tüm parçayı bellekte tutmadan)
        stream.write(WavFile.header(sampleRate, 2, totalFrames * 4L))
        val bytes = ByteArray(block * 4)
        p = 0
        while (p < totalFrames) {
            val n = minOf(block, totalFrames - p)
            mix(p, n, tmp, scale)
            for (i in 0 until n * 2) {
                val v = tmp[i].toInt()
                bytes[2 * i] = (v and 0xff).toByte()
                bytes[2 * i + 1] = ((v shr 8) and 0xff).toByte()
            }
            stream.write(bytes, 0, n * 4)
            p += n
        }
        stream.flush()
    }

    private fun peakOf(start: Int, n: Int, buf: FloatArray): Float {
        java.util.Arrays.fill(buf, 0, n * 2, 0f)
        for (t in tracks) {
            if (t.muted || t.gain <= 0f) continue
            val d = t.data
            for (i in 0 until n) {
                val f = start + i
                if (f >= d.frames) break
                if (d.channels == 1) {
                    val v = d.samples[f] * t.gain
                    buf[2 * i] += v; buf[2 * i + 1] += v
                } else {
                    buf[2 * i] += d.samples[f * d.channels] * t.gain
                    buf[2 * i + 1] += d.samples[f * d.channels + 1] * t.gain
                }
            }
        }
        var m = 0f
        for (i in 0 until n * 2) m = max(m, abs(buf[i]))
        return m
    }
}
