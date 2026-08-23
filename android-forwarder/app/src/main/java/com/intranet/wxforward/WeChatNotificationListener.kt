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
        val pkg = sbn.packageName
        val extras = sbn.notification?.extras ?: return
        val title = extras.getCharSequence(Notification.EXTRA_TITLE)?.toString()?.trim().orEmpty()
        val text = extras.getCharSequence(Notification.EXTRA_TEXT)?.toString()?.trim().orEmpty()

        // 反向：G平台 文件传输助手 → 微信自动回复
        if (Prefs.getReverseEnabled(this) && pkg == Prefs.getTargetPkg(this)) {
            handleReverse(title, text)
            return
        }

        // 正向：微信 → G平台
        if (!Prefs.getEnabled(this)) return
        if (pkg != Prefs.getWechatPkg(this)) return
        handleWeChat(title, text)
    }

    private fun handleWeChat(title: String, text: String) {
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

    /**
     * 处理来自 G平台 的通知，识别模板「回复 张三 好的收到」。
     * 第一阶段：只记录日志、解析出目标和内容，暂不真正发微信。
     */
    private fun handleReverse(title: String, text: String) {
        if (text.isEmpty()) return

        // 诊断：把收到的 G平台 通知都记下来，确认触发是否可行
        LogStore.add("【G平台通知】$title ｜ $text")

        val cleaned = text.replace(Regex("^\\[\\d+\\+?条]"), "").trim()
        // 群/助手通知可能是 "文件传输助手: 回复 张三 内容"，去掉冒号前缀
        val body = run {
            val c = indexOfColon(cleaned)
            if (c in 1..20) cleaned.substring(c + 1).trim() else cleaned
        }

        val keyword = Prefs.getReplyKeyword(this)
        if (!body.startsWith(keyword)) return

        val after = body.removePrefix(keyword).trim()
        val sp = firstSpace(after)
        if (sp <= 0) {
            LogStore.add("⚠️ 指令格式不对，应为「$keyword 对方名字 内容」")
            return
        }
        val target = after.substring(0, sp).trim()
        val content = after.substring(sp + 1).trim()
        if (target.isEmpty() || content.isEmpty()) {
            LogStore.add("⚠️ 指令缺少对方名字或内容")
            return
        }

        // 去重
        val now = System.currentTimeMillis()
        val dedupKey = "R|$target|$content"
        val last = recent[dedupKey]
        if (last != null && now - last < 5000) return
        recent[dedupKey] = now

        LogStore.add("识别到回复指令 → 给【$target】发送：$content（第一阶段仅识别，暂不实际发送）")
        // 第二阶段将在此处把「微信自动回复任务」入队
    }

    private fun firstSpace(s: String): Int {
        val a = s.indexOf(' ')
        val b = s.indexOf('　') // 全角空格
        return when {
            a == -1 -> b
            b == -1 -> a
            else -> minOf(a, b)
        }
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
