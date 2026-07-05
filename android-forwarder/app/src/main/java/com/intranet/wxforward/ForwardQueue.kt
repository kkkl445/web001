package com.intranet.wxforward

import java.util.concurrent.ConcurrentLinkedQueue

/**
 * 连接「通知监听服务」和「无障碍服务」的桥梁。
 * 通知服务把待发消息放进队列；无障碍服务连接后逐条取出、自动发送。
 */
object ForwardQueue {

    data class Task(val text: String, val enqueueAt: Long)

    private val queue = ConcurrentLinkedQueue<Task>()

    /** 无障碍服务连上后把自己注册进来，用于被通知「有新任务了」。 */
    @Volatile
    var pump: (() -> Unit)? = null

    fun enqueue(text: String, now: Long) {
        queue.add(Task(text, now))
        pump?.invoke()
    }

    fun poll(): Task? = queue.poll()

    fun size(): Int = queue.size
}
