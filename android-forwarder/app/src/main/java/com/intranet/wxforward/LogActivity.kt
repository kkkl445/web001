package com.intranet.wxforward

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

class LogActivity : AppCompatActivity() {

    private lateinit var tvLog: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_log)
        tvLog = findViewById(R.id.tvLog)

        findViewById<Button>(R.id.btnClear).setOnClickListener {
            LogStore.clear()
            refresh()
        }
        findViewById<Button>(R.id.btnCopy).setOnClickListener {
            val cm = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            cm.setPrimaryClip(ClipData.newPlainText("log", LogStore.getAllText()))
            Toast.makeText(this, "已复制全部日志", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onResume() {
        super.onResume()
        LogStore.listener = { refresh() }
        refresh()
    }

    override fun onPause() {
        super.onPause()
        LogStore.listener = null
    }

    private fun refresh() {
        tvLog.text = LogStore.getAllText()
    }
}
