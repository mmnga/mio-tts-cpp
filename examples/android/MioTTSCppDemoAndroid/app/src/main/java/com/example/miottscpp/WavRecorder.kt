package com.example.miottscpp

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import kotlin.concurrent.thread
import kotlin.math.max

class WavRecorder(
    val outputFile: File,
    private val sampleRate: Int = 16000,
) {
    @Volatile
    private var running = false

    @Volatile
    private var dataBytesWritten: Long = 0

    private var audioRecord: AudioRecord? = null
    private var worker: Thread? = null

    fun start() {
        check(!running) { "recording is already running" }

        val minBuffer = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        require(minBuffer > 0) { "invalid min buffer size: $minBuffer" }

        val bufferSize = max(minBuffer, sampleRate / 5 * 2)
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize,
        )
        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            recorder.release()
            error("failed to initialize AudioRecord")
        }

        outputFile.parentFile?.mkdirs()
        if (outputFile.exists()) {
            outputFile.delete()
        }

        FileOutputStream(outputFile).use { stream ->
            stream.write(ByteArray(44))
            stream.flush()
        }

        running = true
        dataBytesWritten = 0
        audioRecord = recorder
        recorder.startRecording()

        worker = thread(name = "MioWavRecorder", start = true) {
            FileOutputStream(outputFile, true).use { stream ->
                val buffer = ByteArray(bufferSize)
                while (running) {
                    val n = recorder.read(buffer, 0, buffer.size)
                    if (n > 0) {
                        stream.write(buffer, 0, n)
                        dataBytesWritten += n.toLong()
                    }
                }
                stream.flush()
            }
        }
    }

    fun stop() {
        if (!running) {
            return
        }

        running = false

        val recorder = audioRecord
        try {
            recorder?.stop()
        } catch (_: IllegalStateException) {
        }
        recorder?.release()
        audioRecord = null

        worker?.join(2_000)
        worker = null

        patchWavHeader(outputFile, sampleRate, dataBytesWritten)
    }

    private fun patchWavHeader(file: File, sampleRate: Int, dataSize: Long) {
        RandomAccessFile(file, "rw").use { raf ->
            val channels = 1
            val bitsPerSample = 16
            val byteRate = sampleRate * channels * bitsPerSample / 8
            val blockAlign = channels * bitsPerSample / 8

            raf.seek(0)
            raf.writeBytes("RIFF")
            raf.writeIntLE((36 + dataSize).toInt())
            raf.writeBytes("WAVE")
            raf.writeBytes("fmt ")
            raf.writeIntLE(16)
            raf.writeShortLE(1)
            raf.writeShortLE(channels.toShort())
            raf.writeIntLE(sampleRate)
            raf.writeIntLE(byteRate)
            raf.writeShortLE(blockAlign.toShort())
            raf.writeShortLE(bitsPerSample.toShort())
            raf.writeBytes("data")
            raf.writeIntLE(dataSize.toInt())
        }
    }
}

private fun RandomAccessFile.writeIntLE(value: Int) {
    write(byteArrayOf(
        (value and 0xff).toByte(),
        ((value shr 8) and 0xff).toByte(),
        ((value shr 16) and 0xff).toByte(),
        ((value shr 24) and 0xff).toByte(),
    ))
}

private fun RandomAccessFile.writeShortLE(value: Short) {
    val v = value.toInt() and 0xffff
    write(byteArrayOf(
        (v and 0xff).toByte(),
        ((v shr 8) and 0xff).toByte(),
    ))
}
