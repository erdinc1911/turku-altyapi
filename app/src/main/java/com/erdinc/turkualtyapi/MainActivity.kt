package com.erdinc.turkualtyapi

import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.View
import android.view.WindowManager
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.RadioGroup
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.chaquo.python.Python
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {

    /** Python tarafının ilerleme bildirdiği köprü. */
    inner class ProgressBridge(private val lo: Int, private val hi: Int) {
        fun report(pct: Int, msg: String) {
            val v = lo + (hi - lo) * pct / 100
            runOnUiThread { showProgress(v, msg) }
        }
    }

    private lateinit var pickButton: Button
    private lateinit var generateButton: Button
    private lateinit var regenButton: Button
    private lateinit var playButton: Button
    private lateinit var stopButton: Button
    private lateinit var exportButton: Button
    private lateinit var tempoHalfButton: Button
    private lateinit var tempoDoubleButton: Button
    private lateinit var fileText: TextView
    private lateinit var statusText: TextView
    private lateinit var summaryText: TextView
    private lateinit var timeText: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var resultPanel: View
    private lateinit var mixerPanel: LinearLayout
    private lateinit var positionBar: SeekBar
    private lateinit var densityGroup: RadioGroup
    private lateinit var instrumentBoxes: Map<String, CheckBox>
    private lateinit var tonicSpinner: Spinner
    private lateinit var meterSpinner: Spinner

    private var sourceUri: Uri? = null
    private var projectDir: File? = null
    private var player: StemPlayer? = null
    private var busy = false
    private var userSeeking = false
    private var tempoFactor = 1.0
    private var seed = 1
    private var detectedTonic = 0
    // Başlangıç ses seviyeleri: altyapı kaynağın yaklaşık 8-10 dB altında, sazı bastırmadan.
    private val gains = mutableMapOf(
        "source" to 1.0f, "bas" to 0.9f, "bendir" to 1.0f, "darbuka" to 0.8f,
        "org" to 0.7f, "yaylilar" to 0.4f, "cura" to 0.55f, "ney" to 0.6f
    )
    private val mutes = mutableMapOf<String, Boolean>()

    private val noteNames = listOf("Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si")

    private val pickFile = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            sourceUri = uri
            fileText.text = displayName(uri)
            generateButton.isEnabled = !busy
        }
    }

    private val saveWav = registerForActivityResult(ActivityResultContracts.CreateDocument("audio/wav")) { uri ->
        if (uri != null) exportTo(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        pickButton = findViewById(R.id.pickButton)
        generateButton = findViewById(R.id.generateButton)
        regenButton = findViewById(R.id.regenButton)
        playButton = findViewById(R.id.playButton)
        stopButton = findViewById(R.id.stopButton)
        exportButton = findViewById(R.id.exportButton)
        tempoHalfButton = findViewById(R.id.tempoHalfButton)
        tempoDoubleButton = findViewById(R.id.tempoDoubleButton)
        fileText = findViewById(R.id.fileText)
        statusText = findViewById(R.id.statusText)
        summaryText = findViewById(R.id.summaryText)
        timeText = findViewById(R.id.timeText)
        progressBar = findViewById(R.id.progressBar)
        resultPanel = findViewById(R.id.resultPanel)
        mixerPanel = findViewById(R.id.mixerPanel)
        positionBar = findViewById(R.id.positionBar)
        densityGroup = findViewById(R.id.densityGroup)
        instrumentBoxes = linkedMapOf(
            "bas" to findViewById(R.id.cbBas),
            "bendir" to findViewById(R.id.cbBendir),
            "darbuka" to findViewById(R.id.cbDarbuka),
            "org" to findViewById(R.id.cbOrg),
            "yaylilar" to findViewById(R.id.cbYaylilar),
            "cura" to findViewById(R.id.cbCura),
            "ney" to findViewById(R.id.cbNey),
        )
        tonicSpinner = findViewById(R.id.tonicSpinner)
        meterSpinner = findViewById(R.id.meterSpinner)

        val version = packageManager.getPackageInfo(packageName, 0).versionName
        findViewById<TextView>(R.id.versionText).text = "Sürüm $version · deneme (Aşama 0)"

        tonicSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item,
            listOf("Otomatik") + noteNames)
        meterSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item,
            listOf("Otomatik", "2 vuruş (2/4)", "3 vuruş (3/4)", "4 vuruş (4/4)"))

        pickButton.setOnClickListener { pickFile.launch(arrayOf("audio/*", "video/*")) }
        generateButton.setOnClickListener { startFull() }
        regenButton.setOnClickListener { seed += 1; startArrange() }
        tempoHalfButton.setOnClickListener { tempoFactor = maxOf(0.25, tempoFactor / 2); startArrange() }
        tempoDoubleButton.setOnClickListener { tempoFactor = minOf(4.0, tempoFactor * 2); startArrange() }
        playButton.setOnClickListener { togglePlay() }
        stopButton.setOnClickListener {
            player?.pause(); player?.seekFraction(0f); updatePosition(); playButton.text = "Çal"
        }
        exportButton.setOnClickListener {
            player?.pause(); playButton.text = "Çal"
            val base = fileText.text.toString().substringBeforeLast('.')
            saveWav.launch("${base}_altyapi.wav")
        }
        findViewById<Button>(R.id.licenseButton).setOnClickListener { showLicenses() }
        positionBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar, p: Int, fromUser: Boolean) {
                if (fromUser) { player?.seekFraction(p / 1000f); updateTimeText() }
            }
            override fun onStartTrackingTouch(sb: SeekBar) { userSeeking = true }
            override fun onStopTrackingTouch(sb: SeekBar) { userSeeking = false }
        })
    }

    override fun onDestroy() {
        player?.release()
        super.onDestroy()
    }

    // ------------------------------------------------------------ işlem

    private fun settingsJson(): String {
        val o = JSONObject()
        o.put("density", when (densityGroup.checkedRadioButtonId) {
            R.id.rbSade -> "sade"; R.id.rbZengin -> "zengin"; else -> "dengeli"
        })
        val inst = JSONArray()
        instrumentBoxes.forEach { (key, box) -> if (box.isChecked) inst.put(key) }
        o.put("instruments", inst)
        o.put("tempo_factor", tempoFactor)
        o.put("seed", seed)
        val t = tonicSpinner.selectedItemPosition
        if (t > 0) o.put("tonic_pc", t - 1)
        val m = meterSpinner.selectedItemPosition
        if (m > 0) o.put("beats_per_bar", m + 1)
        return o.toString()
    }

    private fun startFull() {
        val uri = sourceUri ?: return
        tempoFactor = 1.0; seed = 1
        tonicSpinner.setSelection(0); meterSpinner.setSelection(0)
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = File(filesDir, "projeler/$stamp").apply { mkdirs() }
        val settings = settingsJson()
        runJob {
            val t0 = System.currentTimeMillis()
            runOnUiThread { showProgress(0, "Kayıt çözülüyor") }
            AudioDecoder.decodeToWav(this, uri, File(dir, "source.wav")) { pct ->
                runOnUiThread { showProgress(pct * 15 / 100, "Kayıt çözülüyor") }
            }
            val decodeSec = (System.currentTimeMillis() - t0) / 1000.0
            val res = Python.getInstance().getModule("altyapi.pipeline")
                .callAttr("run_full", dir.absolutePath, settings, ProgressBridge(15, 100)).toString()
            projectDir = dir
            Pair(JSONObject(res), decodeSec)
        }
    }

    private fun startArrange() {
        val dir = projectDir ?: return
        if (busy) return
        val settings = settingsJson()
        runJob {
            val res = Python.getInstance().getModule("altyapi.pipeline")
                .callAttr("run_arrange", dir.absolutePath, settings, ProgressBridge(0, 100)).toString()
            Pair(JSONObject(res), null)
        }
    }

    private fun runJob(work: () -> Pair<JSONObject, Double?>) {
        if (busy) return
        setBusy(true)
        player?.pause(); playButton.text = "Çal"
        Thread {
            try {
                val (summary, decodeSec) = work()
                val dir = projectDir!!
                val newPlayer = loadPlayer(dir, summary)
                runOnUiThread {
                    player?.release()
                    player = newPlayer
                    showResult(summary, decodeSec)
                    buildMixer()
                    setBusy(false)
                }
            } catch (t: Throwable) {
                runOnUiThread {
                    setBusy(false)
                    statusText.text = "Hata oluştu."
                    AlertDialog.Builder(this)
                        .setTitle("İşlem tamamlanamadı")
                        .setMessage(t.message ?: t.toString())
                        .setPositiveButton("Tamam", null).show()
                }
            }
        }.start()
    }

    private fun loadPlayer(dir: File, summary: JSONObject): StemPlayer {
        val tracks = mutableListOf(
            StemPlayer.Track("source", "Kaynak kayıt", WavFile.read(File(dir, "source.wav")), gains["source"]!!)
        )
        val stems = summary.optJSONArray("stems") ?: JSONArray()
        for (i in 0 until stems.length()) {
            val s = stems.getJSONObject(i)
            val key = s.getString("key")
            tracks.add(StemPlayer.Track(key, s.getString("name"),
                WavFile.read(File(dir, s.getString("file"))), gains[key] ?: 0.7f))
        }
        tracks.forEach { it.muted = mutes[it.key] ?: false }
        return StemPlayer(tracks)
    }

    // ------------------------------------------------------------ arayüz

    private fun setBusy(b: Boolean) {
        busy = b
        listOf(pickButton, regenButton, tempoHalfButton, tempoDoubleButton, exportButton, playButton, stopButton)
            .forEach { it.isEnabled = !b }
        generateButton.isEnabled = !b && sourceUri != null
        progressBar.visibility = if (b) View.VISIBLE else View.GONE
        if (b) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    private fun showProgress(pct: Int, msg: String) {
        progressBar.progress = pct
        statusText.text = "%$pct · $msg"
    }

    private fun showResult(s: JSONObject, decodeSec: Double?) {
        resultPanel.visibility = View.VISIBLE
        if (!s.optBoolean("ok", true)) {
            statusText.text = s.optString("info", "Ritim bulunamadı")
        } else {
            val timing = StringBuilder()
            decodeSec?.let { timing.append("çözme %.1f sn · ".format(it)) }
            if (!s.isNull("analysis_seconds") && decodeSec != null)
                timing.append("analiz %.1f sn · ".format(s.getDouble("analysis_seconds")))
            timing.append("üretim %.1f sn".format(s.getDouble("render_seconds")))
            statusText.text = "Hazır ($timing)"
        }
        detectedTonic = s.optInt("tonic_pc", 0)
        val cents = s.optDouble("tuning_cents", 0.0)
        val quality = if (s.optString("quality") == "minor") "minör/uşşak benzeri" else "majör benzeri"
        summaryText.text = buildString {
            append("Tempo: ${"%.0f".format(s.optDouble("bpm"))} vuruş/dk\n")
            append("Ölçü: ${s.optInt("beats_per_bar")} vuruş · ${s.optInt("bars")} ölçü\n")
            append("Karar sesi: ${s.optString("tonic_name")} ($quality)\n")
            append("Akort: 440 Hz'e göre ${if (cents >= 0) "+" else ""}${"%.0f".format(cents)} sent ")
            append("(altyapı buna göre akort edildi)\n")
            append("Süre: ${"%.0f".format(s.optDouble("duration"))} sn")
        }
    }

    private fun buildMixer() {
        val p = player ?: return
        mixerPanel.removeAllViews()
        for (t in p.tracks) {
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = android.view.Gravity.CENTER_VERTICAL
            }
            val name = TextView(this).apply {
                text = t.name
                layoutParams = LinearLayout.LayoutParams(dp(120), LinearLayout.LayoutParams.WRAP_CONTENT)
            }
            val mute = CheckBox(this).apply {
                text = "Sustur"
                isChecked = t.muted
                setOnCheckedChangeListener { _, c -> t.muted = c; mutes[t.key] = c }
            }
            val vol = SeekBar(this).apply {
                max = 150
                progress = (t.gain * 100).toInt()
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                    override fun onProgressChanged(sb: SeekBar, v: Int, fromUser: Boolean) {
                        t.gain = v / 100f; gains[t.key] = t.gain
                    }
                    override fun onStartTrackingTouch(sb: SeekBar) {}
                    override fun onStopTrackingTouch(sb: SeekBar) {}
                })
            }
            row.addView(name); row.addView(vol); row.addView(mute)
            mixerPanel.addView(row)
        }
        p.onPosition = { runOnUiThread { updatePosition() } }
        p.onFinished = { runOnUiThread { playButton.text = "Çal" } }
        updatePosition()
    }

    private fun togglePlay() {
        val p = player ?: return
        if (p.isPlaying) { p.pause(); playButton.text = "Çal" }
        else { p.play(); playButton.text = "Duraklat" }
    }

    private fun updatePosition() {
        val p = player ?: return
        if (!userSeeking) positionBar.progress = (1000L * p.position / maxOf(1, p.totalFrames)).toInt()
        updateTimeText()
    }

    private fun updateTimeText() {
        val p = player ?: return
        fun fmt(frames: Int): String {
            val s = frames / p.sampleRate
            return "%d:%02d".format(s / 60, s % 60)
        }
        timeText.text = "${fmt(p.position)} / ${fmt(p.totalFrames)}"
    }

    private fun exportTo(uri: Uri) {
        val p = player ?: return
        setBusy(true)
        showProgress(50, "WAV kaydediliyor")
        Thread {
            try {
                contentResolver.openOutputStream(uri)!!.use { p.exportWav(it.buffered()) }
                runOnUiThread {
                    setBusy(false); statusText.text = "Kaydedildi."
                    Toast.makeText(this, "WAV kaydedildi", Toast.LENGTH_SHORT).show()
                }
            } catch (t: Throwable) {
                runOnUiThread { setBusy(false); statusText.text = "Kaydedilemedi: ${t.message}" }
            }
        }.start()
    }

    private fun displayName(uri: Uri): String {
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) return c.getString(0)
        }
        return uri.lastPathSegment ?: "kayit"
    }

    private fun showLicenses() {
        val text = assets.open("licenses.txt").bufferedReader().use { it.readText() }
        val view = TextView(this).apply {
            this.text = text
            setPadding(dp(20), dp(12), dp(20), dp(12))
            setTextIsSelectable(true)
        }
        AlertDialog.Builder(this)
            .setTitle("Lisanslar ve ses kaynakları")
            .setView(android.widget.ScrollView(this).apply { addView(view) })
            .setPositiveButton("Kapat", null).show()
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()
}
