package com.intranet.wxforward

import android.os.Handler
import android.os.Looper
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 简单的内存日志环形缓冲，用于在 App 里排查转发流程。
 * 失败时无障碍服务会把界面结构 dump 进来，方便定位发送按钮。
 */
object LogStore {
    private const val MAX = 500
    private val lines = ArrayDeque<String>()
    private val fmt = SimpleDateFormat("MM-dd HH:mm:ss", Locale.getDefault())
    private val main = Handler(Looper.getMainLooper())

    /** LogActivity 打开时注册，收到变化就刷新。 */
    @Volatile
    var listener: (() -> Unit)? = null

    @Synchronized
    fun add(msg: String) {
        val line = "${fmt.format(Date())}  $msg"
        lines.addLast(line)
        while (lines.size > MAX) lines.removeFirst()
        val l = listener
        if (l != null) main.post { l.invoke() }
    }

    @Synchronized
    fun getAllText(): String =
        if (lines.isEmpty()) "（暂无日志）" else lines.reversed().joinToString("\n")

    @Synchronized
    fun clear() = lines.clear()
}
