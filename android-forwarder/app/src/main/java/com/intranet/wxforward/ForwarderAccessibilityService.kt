package com.intranet.wxforward

import android.accessibilityservice.AccessibilityService
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo

/**
 * 无障碍服务：把队列里的消息自动发进 G平台 的「文件传输助手」。
 *
 * 流程：打开目标 App → 确保停在文件传输助手会话（必要时点开它）
 *      → 把文字填进输入框 → 找到并点击「发送」。
 * 每一步都有重试；失败时会把当前界面结构 dump 到日志，便于调参。
 */
class ForwarderAccessibilityService : AccessibilityService() {

    private val handler = Handler(Looper.getMainLooper())

    @Volatile
    private var busy = false
    private var currentText: String = ""
    private var targetPkg: String = ""

    // 各阶段重试上限
    private val openChatMaxAttempts = 20
    private val sendMaxAttempts = 10
    private val stepDelay = 500L

    override fun onServiceConnected() {
        super.onServiceConnected()
        ForwardQueue.pump = { handler.post { processNext() } }
        LogStore.add("无障碍服务已连接")
        // 连接时先把积压的任务跑起来
        handler.post { processNext() }
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        // 我们主动轮询界面，不依赖具体事件，这里留空即可。
    }

    override fun onInterrupt() {}

    override fun onDestroy() {
        super.onDestroy()
        if (ForwardQueue.pump != null) ForwardQueue.pump = null
    }

    // ---------------- 主流程 ----------------

    private fun processNext() {
        if (busy) return
        // 队列里的任务一律尝试发送。是否转发微信消息由通知监听端的总开关控制，
        // 这样「发测试消息」即使总开关没开也能验证发送链路。
        val task = ForwardQueue.poll() ?: return
        busy = true
        currentText = task.text
        targetPkg = Prefs.getTargetPkg(this)
        LogStore.add("开始发送：${task.text}")

        copyToClipboard(task.text)

        if (!launchTarget()) {
            fail("打不开目标 App（包名 $targetPkg 不对？请到设置里用「从已安装应用里选择」）")
            return
        }
        ensureInputThenFill(0)
    }

    /** 把目标 App 拉到前台，返回是否成功发起。 */
    private fun launchTarget(): Boolean {
        val launch = packageManager.getLaunchIntentForPackage(targetPkg) ?: return false
        launch.addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
        return try {
            startActivity(launch)
            true
        } catch (t: Throwable) {
            false
        }
    }

    /** 反复检查，直到出现输入框；若在列表页则先点开会话。 */
    private fun ensureInputThenFill(attempt: Int) {
        handler.postDelayed({
            val root = rootInActiveWindow
            if (root == null || root.packageName?.toString() != targetPkg) {
                if (attempt < openChatMaxAttempts) {
                    // 前台被别的 App（如微信）占着时，隔几次重新把目标 App 拉到前台
                    if (attempt % 6 == 5) launchTarget()
                    ensureInputThenFill(attempt + 1)
                } else {
                    fail("目标 App 界面未就绪（当前包名 ${root?.packageName}）", root)
                }
                return@postDelayed
            }

            val edit = findEditable(root)
            if (edit != null) {
                phaseFill(edit)
                return@postDelayed
            }

            // 没有输入框，说明在会话列表页，尝试点开「文件传输助手」
            val chatName = Prefs.getChatName(this)
            val chat = findByText(root, listOf(chatName), requireClickable = false)
            if (chat != null) {
                LogStore.add("点开会话：$chatName")
                clickNodeOrParent(chat)
            }

            if (attempt < openChatMaxAttempts) {
                ensureInputThenFill(attempt + 1)
            } else {
                fail("找不到输入框，也没能打开会话「$chatName」", root)
            }
        }, stepDelay)
    }

    private fun phaseFill(edit: AccessibilityNodeInfo) {
        val ok = setNodeText(edit, currentText)
        LogStore.add(if (ok) "已填入输入框" else "填入输入框失败，尝试粘贴")
        handler.postDelayed({ phaseSend(0) }, stepDelay)
    }

    private fun phaseSend(attempt: Int) {
        val root = rootInActiveWindow
        if (root == null) {
            if (attempt < sendMaxAttempts) handler.postDelayed({ phaseSend(attempt + 1) }, stepDelay)
            else fail("发送阶段拿不到界面")
            return
        }
        val labels = Prefs.getSendLabels(this)
        val send = findByText(root, labels, requireClickable = true)
        if (send != null) {
            val clicked = clickNodeOrParent(send)
            if (clicked) {
                success()
                return
            }
        }
        if (attempt < sendMaxAttempts) {
            handler.postDelayed({ phaseSend(attempt + 1) }, stepDelay)
        } else {
            fail("找不到/点不动发送按钮（尝试的文字：${labels.joinToString("、")}）", root)
        }
    }

    private fun success() {
        LogStore.add("✅ 发送成功")
        finishOne()
    }

    private fun fail(reason: String, root: AccessibilityNodeInfo? = null) {
        LogStore.add("❌ 发送失败：$reason")
        val r = root ?: rootInActiveWindow
        if (r != null) {
            LogStore.add("---- 当前界面结构（发我这段即可调参）----")
            dumpTree(r, 0, StringBuilder()).let { LogStore.add(it) }
            LogStore.add("---- 界面结构结束 ----")
        }
        finishOne()
    }

    private fun finishOne() {
        busy = false
        // 稍等再处理下一条，给界面留出缓冲
        handler.postDelayed({ processNext() }, 1000)
    }

    // ---------------- 界面查找工具 ----------------

    private fun findEditable(node: AccessibilityNodeInfo?): AccessibilityNodeInfo? {
        node ?: return null
        if (node.isEditable ||
            node.className?.toString()?.contains("EditText") == true
        ) {
            return node
        }
        for (i in 0 until node.childCount) {
            val r = findEditable(node.getChild(i))
            if (r != null) return r
        }
        return null
    }

    private fun findByText(
        node: AccessibilityNodeInfo?,
        candidates: List<String>,
        requireClickable: Boolean
    ): AccessibilityNodeInfo? {
        node ?: return null
        val text = node.text?.toString()?.trim()
        val desc = node.contentDescription?.toString()?.trim()
        val matched = candidates.any { c ->
            c.isNotEmpty() && (text == c || desc == c ||
                    (text != null && text.length <= 6 && text.contains(c)) ||
                    (desc != null && desc.length <= 6 && desc.contains(c)))
        }
        if (matched) {
            if (!requireClickable || node.isClickable || hasClickableAncestor(node)) {
                return node
            }
        }
        for (i in 0 until node.childCount) {
            val r = findByText(node.getChild(i), candidates, requireClickable)
            if (r != null) return r
        }
        return null
    }

    private fun hasClickableAncestor(node: AccessibilityNodeInfo?): Boolean {
        var p = node?.parent
        var depth = 0
        while (p != null && depth < 6) {
            if (p.isClickable) return true
            p = p.parent
            depth++
        }
        return false
    }

    private fun clickNodeOrParent(node: AccessibilityNodeInfo?): Boolean {
        var n = node
        var depth = 0
        while (n != null && depth < 6) {
            if (n.isClickable) {
                return n.performAction(AccessibilityNodeInfo.ACTION_CLICK)
            }
            n = n.parent
            depth++
        }
        // 兜底：直接对原节点发点击
        return node?.performAction(AccessibilityNodeInfo.ACTION_CLICK) ?: false
    }

    private fun setNodeText(node: AccessibilityNodeInfo, text: String): Boolean {
        node.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
        val args = Bundle()
        args.putCharSequence(
            AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text
        )
        val ok = node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
        if (ok) return true
        // 兜底：粘贴剪贴板
        return node.performAction(AccessibilityNodeInfo.ACTION_PASTE)
    }

    private fun copyToClipboard(text: String) {
        try {
            val cm = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            cm.setPrimaryClip(ClipData.newPlainText("wxforward", text))
        } catch (_: Throwable) {
        }
    }

    private fun dumpTree(node: AccessibilityNodeInfo?, depth: Int, sb: StringBuilder): String {
        node ?: return sb.toString()
        if (depth > 25) return sb.toString()
        val indent = "  ".repeat(depth)
        val cls = node.className?.toString()?.substringAfterLast('.') ?: "?"
        val text = node.text?.toString()?.take(20)
        val desc = node.contentDescription?.toString()?.take(20)
        val id = node.viewIdResourceName?.substringAfterLast('/')
        val flags = buildString {
            if (node.isClickable) append("C")
            if (node.isEditable) append("E")
        }
        sb.append(indent).append(cls)
        if (!id.isNullOrEmpty()) sb.append(" #").append(id)
        if (!text.isNullOrEmpty()) sb.append(" \"").append(text).append("\"")
        if (!desc.isNullOrEmpty()) sb.append(" desc=").append(desc)
        if (flags.isNotEmpty()) sb.append(" [").append(flags).append("]")
        sb.append("\n")
        for (i in 0 until node.childCount) {
            dumpTree(node.getChild(i), depth + 1, sb)
        }
        return sb.toString()
    }
}
