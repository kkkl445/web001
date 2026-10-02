import json, numpy as np, soundfile as sf
from script import SCENES, tts_text
from tts import tts
LEAD, GAP, TAIL = 0.7, 0.45, 0.9
# 每个场景动画所需的最短时长（相对最后一句开始的补充由渲染端决定）
MIN_EXTRA = {'p': 0, 'bang': 0}
SR = None; chunks = []; timeline = []; t = 0.0
for sc in SCENES:
    segs = []; st = t
    cur = LEAD
    pieces = [np.zeros(int(LEAD*44100), np.float32)]
    for i, line in enumerate(sc['lines']):
        a = tts.generate(tts_text(line), sid=0, speed=1.05)
        SR = a.sample_rate
        samp = np.array(a.samples, np.float32)
        d = len(samp)/SR
        segs.append([cur, cur+d, line])
        pieces.append(samp); pieces.append(np.zeros(int(GAP*SR), np.float32))
        cur += d + GAP
    timeline.append(dict(id=sc['id'], name=sc['name'], start=st, segs=segs, audio_len=cur+TAIL-GAP))
    chunks.append(pieces)
    t += cur + TAIL - GAP
    print(sc['id'], round(cur,2))
json.dump(dict(sr=SR, scenes=timeline), open('timeline_raw.json','w'), ensure_ascii=False, indent=1)
np.save('chunks.npy', np.array([np.concatenate(p) for p in chunks], dtype=object), allow_pickle=True)
