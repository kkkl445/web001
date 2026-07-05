package com.intranet.wxforward

import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

class WhitelistActivity : AppCompatActivity() {

    private lateinit var container: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_whitelist)
        container = findViewById(R.id.container)
        val etName = findViewById<EditText>(R.id.etName)

        findViewById<Button>(R.id.btnAdd).setOnClickListener {
            val name = etName.text.toString().trim()
            if (name.isEmpty()) {
                Toast.makeText(this, "请输入名字", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val set = Prefs.getWhitelist(this)
            set.add(name)
            Prefs.setWhitelist(this, set)
            etName.setText("")
            render()
        }
        render()
    }

    private fun render() {
        container.removeAllViews()
        val set = Prefs.getWhitelist(this).sorted()
        if (set.isEmpty()) {
            val tv = TextView(this)
            tv.text = "还没有添加联系人"
            tv.setPadding(0, 24, 0, 0)
            tv.setTextColor(0xFF999999.toInt())
            container.addView(tv)
            return
        }
        for (name in set) {
            container.addView(buildRow(name))
        }
    }

    private fun buildRow(name: String): LinearLayout {
        val row = LinearLayout(this)
        row.orientation = LinearLayout.HORIZONTAL
        row.gravity = Gravity.CENTER_VERTICAL
        row.setBackgroundColor(0xFFFFFFFF.toInt())
        row.setPadding(32, 28, 32, 28)
        val lp = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        )
        lp.bottomMargin = 12
        row.layoutParams = lp

        val tv = TextView(this)
        tv.text = name
        tv.textSize = 16f
        tv.setTextColor(0xFF222222.toInt())
        val tvLp = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        tv.layoutParams = tvLp
        row.addView(tv)

        val del = Button(this)
        del.text = "删除"
        del.setOnClickListener {
            val set = Prefs.getWhitelist(this)
            set.remove(name)
            Prefs.setWhitelist(this, set)
            render()
        }
        row.addView(del)
        return row
    }
}
