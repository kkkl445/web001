package com.intranet.wxforward

import android.app.Notification
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification

/**
 * 读取微信通知，按白名单过滤发件人，把「发件人+内容」放进转发队列。
 */
class WeChatNotificationListener : NotificationListenerService() {

    // 简单去重：记录最近处理过的通知，避免微信刷新导致重复转发。
    private val recent = object : LinkedHashMap<String, Long>(64, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, Long>?): Boolean =
            size > 200
    }

    override fun onNotificationPosted(sbn: StatusBarNotification?) {
        sbn ?: return
        try {
            handle(sbn)
        } catch (t: Throwable) {
            LogStore.add("处理通知异常：${t.message}")
        }
    }

    private fun handle(sbn: StatusBarNotification) {
        if (!Prefs.getEnabled(this)) return

        val wechatPkg = Prefs.getWechatPkg(this)
        if (sbn.packageName != wechatPkg) return

        val extras = sbn.notification?.extras ?: return
        val title = extras.getCharSequence(Notification.EXTRA_TITLE)?.toString()?.trim().orEmpty()
        val text = extras.getCharSequence(Notification.EXTRA_TEXT)?.toString()?.trim().orEmpty()

        if (title.isEmpty() || text.isEmpty()) return

        // 过滤微信的汇总通知，例如标题「微信」内容「[3条]xxx」
        if (title == "微信" && text.contains("条")) return

        // 去掉微信未读计数前缀，如 "[2条]" "[99+条]"
        val cleaned = text.replace(Regex("^\\[\\d+\\+?条]"), "").trim()

        // 群消息内容常见格式 "张三: 你好"，此时真正的发件人在冒号前。
        var sender = title
        var content = cleaned
        val colon = indexOfColon(cleaned)
        if (colon in 1..20) {
            val maybeSender = cleaned.substring(0, colon).trim()
            val maybeContent = cleaned.substring(colon + 1).trim()
            if (maybeSender.isNotEmpty() && maybeContent.isNotEmpty()) {
                content = maybeContent
                // 群聊：标题是群名，发件人在内容里；单聊带计数时发件人=标题，不重复
                sender = if (maybeSender == title) title else "$title / $maybeSender"
            }
        }

        val whitelist = Prefs.getWhitelist(this)
        if (whitelist.isEmpty()) return
        val hit = whitelist.any { key ->
            key.isNotBlank() && (title.contains(key) || sender.contains(key))
        }
        if (!hit) {
            LogStore.add("忽略（不在白名单）：$title")
            return
        }

        // 去重
        val now = System.currentTimeMillis()
        val dedupKey = "$title|$text"
        val last = recent[dedupKey]
        if (last != null && now - last < 5000) return
        recent[dedupKey] = now

        val msg = Prefs.formatMessage(this, sender, content)
        LogStore.add("命中白名单，入队：$msg")
        ForwardQueue.enqueue(msg, now)
    }

    private fun indexOfColon(s: String): Int {
        val a = s.indexOf('：') // 中文冒号
        val b = s.indexOf(':')  // 英文冒号
        return when {
            a == -1 -> b
            b == -1 -> a
            else -> minOf(a, b)
        }
    }
}
