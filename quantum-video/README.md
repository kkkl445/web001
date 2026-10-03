# 量子力学 · Quantum Mechanics

A 3½-minute (208 s), 1080p / 30 fps popular-science film about quantum mechanics.
It has no voiceover: on-screen typography carries the story, with an original BGM underneath.

**成片：[`quantum-mechanics.mp4`](quantum-mechanics.mp4)**

## 内容

| # | 章节 | 画面 |
|---|------|------|
| — | 开场 | 原子晶格无限放大，中心原子化作电子云，片名浮现 |
| 01 | 量子化 | 连续斜坡“折”成台阶，粒子逐级跃迁；E = hν |
| 02 | 波粒二象性 | 电子逐个穿过双缝，单点累积成干涉条纹，波纹场显现 |
| 03 | 测量 | 探测器开启，条纹消失，分布变为平滑的包络 |
| 04 | 叠加与坍缩 | 3d 轨道概率云（采样自真实氢原子波函数）坍缩为一个点；薛定谔方程 |
| 05 | 不确定性原理 | 位置/动量分布此消彼长；Δx·Δp ≥ ħ/2 |
| 06 | 量子隧穿 | 经典小球被弹回；量子波包的分步傅里叶数值模拟，算出约 21% 透射 |
| 07 | 量子纠缠 | 一对粒子分离，测量 A 的瞬间 B 的自旋随之确定 |
| 08 | 无处不在 | 芯片、激光、核磁共振、原子钟 → 量子计算、量子通信 |
| — | 尾声 | 费曼名言；“世界的底层，远比我们想象的更奇妙。” |

## 风格

- 深墨蓝背景、暗角与细颗粒；象牙白文字、香槟金强调、冰蓝/紫色光效
- 思源宋体（Noto Serif SC）标题与正文，Cormorant Garamond 斜体英文副标，Jost 字距拉开的小标签，STIX Two 排公式
- 左栏逐字浮现的叙述、右侧物理动画、底部极简进度条

## BGM

原创配乐，全部由 `music.py` 用代码合成，不使用任何采样或第三方音乐，没有版权问题。

- D 小调，60 BPM，每 8 秒换一次和弦（Dm9 – B♭maj9 – Fmaj9 – C6/9 …，结尾落在 Dsus2）
- 音色：温暖的失谐 Pad、低音、毛毡钢琴、FM 玻璃钟、高八度微光、空气声，加一个合成的立体声大厅混响
- 与画面同步：片名、坍缩、纠缠测量、第 8 章抬升、结尾句各有一次低频重音或钟声；整体响度随章节起伏，响度 −17 LUFS

## 重新生成

```bash
pip install -r requirements.txt   # 另需 ffmpeg
./build.sh                        # 下载字体 → 物理预计算 → 配乐 → 逐帧渲染 → 合成
python3 render.py --preview 12,58,97 --out /tmp/stills   # 只渲染几张静帧预览
```

| 文件 | 作用 |
|------|------|
| `timeline.py` | 场景时长与音画同步点（画面和配乐共用） |
| `engine.py` | numpy/OpenCV/Pillow 合成引擎：抗锯齿线条、辉光、逐字动画排版 |
| `physics.py` | 氢原子轨道采样、双缝统计、隧穿数值模拟（结果缓存在 `.cache/`） |
| `scenes.py` | 十个场景 |
| `render.py` | 多进程渲染 + x264 编码 |
| `music.py` | 配乐合成与母带处理 |

字体均为 SIL OFL 授权，由 `fetch_fonts.sh` 从 google/fonts 下载。
