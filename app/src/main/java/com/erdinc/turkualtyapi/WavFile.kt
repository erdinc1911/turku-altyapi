package com.erdinc.turkualtyapi

import java.io.File
import java.io.OutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** 16-bit PCM WAV okuma/yazma yardımcıları. */
object WavFile {

    class Data(val sampleRate: Int, val channels: Int, val samples: ShortArray) {
        val frames: Int get() = samples.size / channels
    }

    fun header(sampleRate: Int, channels: Int, dataBytes: Long): ByteArray {
        val b = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        b.put("RIFF".toByteArray()); b.putInt((36 + dataBytes).toInt())
        b.put("WAVE".toByteArray()); b.put("fmt ".toByteArray())
        b.putInt(16); b.putShort(1); b.putShort(channels.toShort())
        b.putInt(sampleRate); b.putInt(sampleRate * channels * 2)
        b.putShort((channels * 2).toShort()); b.putShort(16)
        b.put("data".toByteArray()); b.putInt(dataBytes.toInt())
        return b.array()
    }

    /** Başlığı sona yazılmak üzere boş bırakılan akış yazıcısı. */
    class Writer(file: File) {
        private val raf = RandomAccessFile(file, "rw")
        private var bytes = 0L

        init {
            raf.setLength(0)
            raf.write(ByteArray(44))
        }

        fun write(buf: ByteArray, len: Int) {
            raf.write(buf, 0, len)
            bytes += len
        }

        fun close(sampleRate: Int, channels: Int) {
            raf.seek(0)
            raf.write(header(sampleRate, channels, bytes))
            raf.close()
        }
    }

    fun read(file: File): Data {
        RandomAccessFile(file, "r").use { raf ->
            val all = ByteArray(raf.length().toInt())
            raf.readFully(all)
            val bb = ByteBuffer.wrap(all).order(ByteOrder.LITTLE_ENDIAN)
            var pos = 12
            var sr = 44100
            var ch = 1
            while (pos + 8 <= all.size) {
                val id = String(all, pos, 4)
                val size = bb.getInt(pos + 4)
                if (id == "fmt ") {
                    ch = bb.getShort(pos + 10).toInt()
                    sr = bb.getInt(pos + 12)
                } else if (id == "data") {
                    val n = minOf(size, all.size - pos - 8) / 2
                    val out = ShortArray(n)
                    bb.position(pos + 8)
                    bb.asShortBuffer().get(out, 0, n)
                    return Data(sr, ch, out)
                }
                pos += 8 + size + (size and 1)
            }
            throw IllegalArgumentException("WAV veri bölümü bulunamadı: ${file.name}")
        }
    }

    fun writeStream(out: OutputStream, sampleRate: Int, channels: Int, pcm: ShortArray) {
        out.write(header(sampleRate, channels, pcm.size * 2L))
        val buf = ByteBuffer.allocate(64 * 1024).order(ByteOrder.LITTLE_ENDIAN)
        var i = 0
        while (i < pcm.size) {
            buf.clear()
            while (i < pcm.size && buf.remaining() >= 2) buf.putShort(pcm[i++])
            out.write(buf.array(), 0, buf.position())
        }
    }
}
