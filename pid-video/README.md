# PID 控制算法科普视频

- `PID科普.mp4`：成品视频（1280×720，约 3 分 12 秒，中文配音 + 内嵌字幕）
- `PID科普.srt`：独立字幕文件

## 内容大纲
1. 引入：无人机悬停、空调恒温、平衡车
2. 问题建模：误差 e = 目标 − 实际，闭环反馈
3. 开关控制：只看方向不看大小 → 来回振荡
4. P 比例控制：像弹簧；稳态误差 e = mg/Kp；Kp 过大会振荡
5. I 积分控制：累积误差消除稳态误差，但会超调
6. D 微分控制：看变化速度，像减震器，抑制超调
7. PID 公式与四种方案对比（P 看现在、I 看过去、D 看未来）
8. 代码实现、调参经验、积分饱和与微分噪声
9. 应用场景与结尾

视频中的曲线均来自真实的无人机高度仿真（`src/sim.py`），不是手绘示意。

## 重新生成
依赖：`numpy pillow sherpa-onnx soundfile` 与 `ffmpeg`，以及离线语音模型
[vits-melo-tts-zh_en](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2)（解压到 `src/` 目录下）。

```bash
cd src
python3 build_audio.py   # 合成旁白
python3 make.py          # 渲染画面并合成 mp4 / srt
```
