package com.intranet.wxforward

import android.content.Intent
import android.os.Bundle
import android.provider.Settings
import android.text.TextUtils
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat

class MainActivity : AppCompatActivity() {

    private lateinit var tvStatus: TextView
    private lateinit var switchEnabled: SwitchCompat
    private lateinit var switchReverse: SwitchCompat

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvStatus = findViewById(R.id.tvStatus)
        switchEnabled = findViewById(R.id.switchEnabled)
        switchReverse = findViewById(R.id.switchReverse)

        switchEnabled.setOnCheckedChangeListener { _, checked ->
            Prefs.setEnabled(this, checked)
            LogStore.add(if (checked) "转发已开启" else "转发已关闭")
        }
        switchReverse.setOnCheckedChangeListener { _, checked ->
            Prefs.setReverseEnabled(this, checked)
            LogStore.add(if (checked) "反向自动回复已开启" else "反向自动回复已关闭")
        }

        findViewById<Button>(R.id.btnNotifPerm).setOnClickListener {
            startActivity(Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS))
        }
        findViewById<Button>(R.id.btnA11yPerm).setOnClickListener {
            startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
        }
        findViewById<Button>(R.id.btnWhitelist).setOnClickListener {
            startActivity(Intent(this, WhitelistActivity::class.java))
        }
        findViewById<Button>(R.id.btnSettings).setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }
        findViewById<Button>(R.id.btnLog).setOnClickListener {
            startActivity(Intent(this, LogActivity::class.java))
        }
        findViewById<Button>(R.id.btnTest).setOnClickListener {
            val msg = Prefs.formatMessage(this, "测试联系人", "这是一条测试消息")
            ForwardQueue.enqueue(msg, System.currentTimeMillis())
            LogStore.add("已手动入队一条测试消息")
            Toast.makeText(this, "已入队，去看运行日志", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onResume() {
        super.onResume()
        switchEnabled.isChecked = Prefs.getEnabled(this)
        switchReverse.isChecked = Prefs.getReverseEnabled(this)
        refreshStatus()
    }

    private fun refreshStatus() {
        val notifOk = isNotificationListenerEnabled()
        val a11yOk = isAccessibilityEnabled()
        val whitelist = Prefs.getWhitelist(this)

        val version = try {
            packageManager.getPackageInfo(packageName, 0).versionName
        } catch (_: Exception) {
            "?"
        }

        val sb = StringBuilder()
        sb.append("📌 当前版本：$version\n")
        sb.append(if (notifOk) "✅ 微信通知读取权限：已开启\n" else "❌ 微信通知读取权限：未开启（点①）\n")
        sb.append(if (a11yOk) "✅ 无障碍自动发送权限：已开启\n" else "❌ 无障碍自动发送权限：未开启（点②）\n")
        sb.append("白名单联系人：${whitelist.size} 个\n")
        sb.append("目标 App 包名：${Prefs.getTargetPkg(this)}\n")
        sb.append("目标会话：${Prefs.getChatName(this)}")
        tvStatus.text = sb.toString()
    }

    private fun isNotificationListenerEnabled(): Boolean {
        val flat = Settings.Secure.getString(contentResolver, "enabled_notification_listeners")
        if (TextUtils.isEmpty(flat)) return false
        return flat.split(":").any { it.contains(packageName) }
    }

    private fun isAccessibilityEnabled(): Boolean {
        val flat = Settings.Secure.getString(
            contentResolver, Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES
        ) ?: return false
        val target = "$packageName/${ForwarderAccessibilityService::class.java.name}"
        return flat.split(":").any { it.equals(target, ignoreCase = true) || it.contains(packageName) }
    }
}
