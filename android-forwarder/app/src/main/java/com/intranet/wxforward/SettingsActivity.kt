package com.intranet.wxforward

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
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

        findViewById<Button>(R.id.btnPickApp).setOnClickListener {
            showAppPicker { pkg -> etTargetPkg.setText(pkg) }
        }

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

    /** 列出所有可启动的已安装应用，让用户点选目标 App。 */
    private fun showAppPicker(onPick: (String) -> Unit) {
        val pm = packageManager
        val intent = Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER)
        val acts = pm.queryIntentActivities(intent, 0)
        val apps = acts
            .map { ri ->
                val label = ri.loadLabel(pm).toString()
                val pkg = ri.activityInfo.packageName
                label to pkg
            }
            .distinctBy { it.second }
            .sortedBy { it.first.lowercase() }

        if (apps.isEmpty()) {
            Toast.makeText(this, "读取应用列表失败", Toast.LENGTH_SHORT).show()
            return
        }

        val display = apps.map { "${it.first}\n${it.second}" }.toTypedArray()
        AlertDialog.Builder(this)
            .setTitle("选择目标 App（找 G平台）")
            .setItems(display) { _, which ->
                val pkg = apps[which].second
                onPick(pkg)
                Toast.makeText(this, "已选择：$pkg", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("取消", null)
            .show()
    }
}
