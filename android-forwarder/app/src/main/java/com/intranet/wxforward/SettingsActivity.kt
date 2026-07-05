package com.intranet.wxforward

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        val etTargetPkg = findViewById<EditText>(R.id.etTargetPkg)
        val etChatName = findViewById<EditText>(R.id.etChatName)
        val etSendLabels = findViewById<EditText>(R.id.etSendLabels)
        val etTemplate = findViewById<EditText>(R.id.etTemplate)
        val etWechatPkg = findViewById<EditText>(R.id.etWechatPkg)

        fun load() {
            etTargetPkg.setText(Prefs.getTargetPkg(this))
            etChatName.setText(Prefs.getChatName(this))
            etSendLabels.setText(Prefs.getSendLabelsRaw(this))
            etTemplate.setText(Prefs.getTemplate(this))
            etWechatPkg.setText(Prefs.getWechatPkg(this))
        }
        load()

        findViewById<Button>(R.id.btnSave).setOnClickListener {
            Prefs.setTargetPkg(this, etTargetPkg.text.toString())
            Prefs.setChatName(this, etChatName.text.toString())
            Prefs.setSendLabels(this, etSendLabels.text.toString())
            Prefs.setTemplate(this, etTemplate.text.toString())
            Prefs.setWechatPkg(this, etWechatPkg.text.toString())
            Toast.makeText(this, "已保存", Toast.LENGTH_SHORT).show()
            finish()
        }

        findViewById<Button>(R.id.btnReset).setOnClickListener {
            etTargetPkg.setText(Prefs.DEFAULT_TARGET_PKG)
            etChatName.setText(Prefs.DEFAULT_CHAT_NAME)
            etSendLabels.setText(Prefs.DEFAULT_SEND_LABELS)
            etTemplate.setText(Prefs.DEFAULT_TEMPLATE)
            etWechatPkg.setText(Prefs.DEFAULT_WECHAT_PKG)
            Toast.makeText(this, "已恢复默认，记得点保存", Toast.LENGTH_SHORT).show()
        }
    }
}
