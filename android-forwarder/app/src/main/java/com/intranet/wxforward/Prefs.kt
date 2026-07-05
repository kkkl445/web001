package com.intranet.wxforward

import android.content.Context

/**
 * 所有可配置项统一放这里，用 SharedPreferences 持久化。
 */
object Prefs {
    private const val FILE = "wx_forward_prefs"

    private const val KEY_ENABLED = "enabled"
    private const val KEY_WHITELIST = "whitelist"
    private const val KEY_TARGET_PKG = "target_pkg"
    private const val KEY_CHAT_NAME = "chat_name"
    private const val KEY_SEND_LABELS = "send_labels"
    private const val KEY_TEMPLATE = "template"
    private const val KEY_WECHAT_PKG = "wechat_pkg"

    // G平台（云之家二次开发）默认包名。若不对，可在设置里改。
    const val DEFAULT_TARGET_PKG = "com.yunzhijia.gree"
    const val DEFAULT_WECHAT_PKG = "com.tencent.mm"
    const val DEFAULT_CHAT_NAME = "文件传输助手"
    const val DEFAULT_SEND_LABELS = "发送,Send,发 送"
    const val DEFAULT_TEMPLATE = "【微信】{sender}：{content}"

    private fun sp(ctx: Context) =
        ctx.applicationContext.getSharedPreferences(FILE, Context.MODE_PRIVATE)

    fun getEnabled(ctx: Context) = sp(ctx).getBoolean(KEY_ENABLED, false)
    fun setEnabled(ctx: Context, v: Boolean) = sp(ctx).edit().putBoolean(KEY_ENABLED, v).apply()

    /** 白名单：发件人名字集合（匹配微信通知标题）。 */
    fun getWhitelist(ctx: Context): MutableSet<String> {
        val raw = sp(ctx).getStringSet(KEY_WHITELIST, emptySet()) ?: emptySet()
        return raw.toMutableSet()
    }

    fun setWhitelist(ctx: Context, set: Set<String>) {
        sp(ctx).edit().putStringSet(KEY_WHITELIST, set).apply()
    }

    fun getTargetPkg(ctx: Context): String =
        sp(ctx).getString(KEY_TARGET_PKG, DEFAULT_TARGET_PKG)!!.ifBlank { DEFAULT_TARGET_PKG }

    fun setTargetPkg(ctx: Context, v: String) =
        sp(ctx).edit().putString(KEY_TARGET_PKG, v.trim()).apply()

    fun getWechatPkg(ctx: Context): String =
        sp(ctx).getString(KEY_WECHAT_PKG, DEFAULT_WECHAT_PKG)!!.ifBlank { DEFAULT_WECHAT_PKG }

    fun setWechatPkg(ctx: Context, v: String) =
        sp(ctx).edit().putString(KEY_WECHAT_PKG, v.trim()).apply()

    fun getChatName(ctx: Context): String =
        sp(ctx).getString(KEY_CHAT_NAME, DEFAULT_CHAT_NAME)!!.ifBlank { DEFAULT_CHAT_NAME }

    fun setChatName(ctx: Context, v: String) =
        sp(ctx).edit().putString(KEY_CHAT_NAME, v.trim()).apply()

    /** 发送按钮可能的文字，逗号分隔。 */
    fun getSendLabels(ctx: Context): List<String> =
        sp(ctx).getString(KEY_SEND_LABELS, DEFAULT_SEND_LABELS)!!
            .split(",").map { it.trim() }.filter { it.isNotEmpty() }

    fun setSendLabels(ctx: Context, v: String) =
        sp(ctx).edit().putString(KEY_SEND_LABELS, v.trim()).apply()

    fun getSendLabelsRaw(ctx: Context): String =
        sp(ctx).getString(KEY_SEND_LABELS, DEFAULT_SEND_LABELS)!!

    fun getTemplate(ctx: Context): String =
        sp(ctx).getString(KEY_TEMPLATE, DEFAULT_TEMPLATE)!!.ifBlank { DEFAULT_TEMPLATE }

    fun setTemplate(ctx: Context, v: String) =
        sp(ctx).edit().putString(KEY_TEMPLATE, v).apply()

    fun formatMessage(ctx: Context, sender: String, content: String): String =
        getTemplate(ctx)
            .replace("{sender}", sender)
            .replace("{content}", content)
}
