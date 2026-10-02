import subprocess, json, numpy as np, soundfile as sf
from multiprocessing import Pool
import render
N = int(render.TOTAL * render.FPS)
# 音频：按最终场景时长拼接
raw = json.load(open('timeline_raw.json')); sr = raw['sr']
chunks = np.load('chunks.npy', allow_pickle=True)
audio = np.zeros(int(render.TOTAL * sr) + sr, np.float32)
for sc, ch in zip(render.SC, chunks):
    i = int(sc['start'] * sr); audio[i:i + len(ch)] += ch
audio = audio / max(1e-6, np.abs(audio).max()) * 0.9
sf.write('narration.wav', audio, sr)
# 字幕 SRT
def ts(x): h=int(x//3600); m=int(x%3600//60); s=x%60; return f'{h:02d}:{m:02d}:{int(s):02d},{int((s%1)*1000):03d}'
k=1; out=[]
for sc in render.SC:
    for a,b,l in sc['segs']:
        out.append(f'{k}\n{ts(sc["start"]+a)} --> {ts(sc["start"]+b+0.3)}\n{l}\n'); k+=1
open('PID科普.srt','w').write('\n'.join(out))
def f(n): return render.frame(n).tobytes()
p = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-',
    '-i','narration.wav','-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k',
    '-shortest','-movflags','+faststart','PID科普.mp4'], stdin=subprocess.PIPE)
with Pool(8) as pool:
    for k, b in enumerate(pool.imap(f, range(N), chunksize=16)):
        p.stdin.write(b)
        if k % 900 == 0: print(k, '/', N, flush=True)
p.stdin.close(); p.wait()
