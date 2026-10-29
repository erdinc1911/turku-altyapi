package com.erdinc.turkualtyapi

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.io.File
import java.nio.ByteOrder

/**
 * Ses veya video dosyasındaki ilk ses izini Android'in kendi çözücüleriyle
 * 16-bit PCM WAV'a çevirir (FFmpeg gerekmez). En fazla 2 kanal tutulur.
 */
object AudioDecoder {

    class Result(val sampleRate: Int, val channels: Int, val durationSec: Double)

    fun decodeToWav(ctx: Context, uri: Uri, out: File, onProgress: (Int) -> Unit): Result {
        val ex = MediaExtractor()
        ex.setDataSource(ctx, uri, null)
        var track = -1
        for (i in 0 until ex.trackCount) {
            val mime = ex.getTrackFormat(i).getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("audio/")) { track = i; break }
        }
        require(track >= 0) { "Dosyada ses izi bulunamadı" }
        ex.selectTrack(track)
        val inFormat = ex.getTrackFormat(track)
        val mime = inFormat.getString(MediaFormat.KEY_MIME)!!
        val durationUs = if (inFormat.containsKey(MediaFormat.KEY_DURATION))
            inFormat.getLong(MediaFormat.KEY_DURATION) else 0L

        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(inFormat, null, null, 0)
        codec.start()

        var sampleRate = inFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        var srcChannels = inFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        var floatPcm = false
        val writer = WavFile.Writer(out)
        val info = MediaCodec.BufferInfo()
        var inputDone = false
        var outputDone = false
        var lastPct = -1
        var outBytes = ByteArray(0)

        try {
            while (!outputDone) {
                if (!inputDone) {
                    val inIdx = codec.dequeueInputBuffer(10_000)
                    if (inIdx >= 0) {
                        val buf = codec.getInputBuffer(inIdx)!!
                        val n = ex.readSampleData(buf, 0)
                        if (n < 0) {
                            codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(inIdx, 0, n, ex.sampleTime, 0)
                            ex.advance()
                        }
                    }
                }
                val outIdx = codec.dequeueOutputBuffer(info, 10_000)
                when {
                    outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        val f = codec.outputFormat
                        sampleRate = f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        srcChannels = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                        floatPcm = f.containsKey(MediaFormat.KEY_PCM_ENCODING) &&
                            f.getInteger(MediaFormat.KEY_PCM_ENCODING) == AudioFormat.ENCODING_PCM_FLOAT
                    }
                    outIdx >= 0 -> {
                        if (info.size > 0) {
                            val buf = codec.getOutputBuffer(outIdx)!!
                            buf.position(info.offset)
                            buf.limit(info.offset + info.size)
                            buf.order(ByteOrder.LITTLE_ENDIAN)
                            val outCh = minOf(2, srcChannels)
                            val bytesPerSample = if (floatPcm) 4 else 2
                            val frames = info.size / (bytesPerSample * srcChannels)
                            val need = frames * outCh * 2
                            if (outBytes.size < need) outBytes = ByteArray(need)
                            var o = 0
                            for (fr in 0 until frames) {
                                for (c in 0 until srcChannels) {
                                    val s: Short = if (floatPcm) {
                                        val v = buf.float.coerceIn(-1f, 1f)
                                        (v * 32767f).toInt().toShort()
                                    } else buf.short
                                    if (c < outCh) {
                                        outBytes[o++] = (s.toInt() and 0xff).toByte()
                                        outBytes[o++] = ((s.toInt() shr 8) and 0xff).toByte()
                                    }
                                }
                            }
                            writer.write(outBytes, o)
                            if (durationUs > 0) {
                                val pct = (100 * info.presentationTimeUs / durationUs).toInt().coerceIn(0, 100)
                                if (pct != lastPct) { lastPct = pct; onProgress(pct) }
                            }
                        }
                        codec.releaseOutputBuffer(outIdx, false)
                        if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
                    }
                }
            }
        } finally {
            writer.close(sampleRate, minOf(2, srcChannels))
            codec.stop(); codec.release(); ex.release()
        }
        return Result(sampleRate, minOf(2, srcChannels), durationUs / 1e6)
    }
}
