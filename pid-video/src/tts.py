import sherpa_onnx, soundfile as sf, sys
d="vits-melo-tts-zh_en/"
cfg=sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=d+"model.onnx",lexicon=d+"lexicon.txt",tokens=d+"tokens.txt",dict_dir=d+"dict"),num_threads=4),rule_fsts=d+"date.fst,"+d+"phone.fst,"+d+"number.fst")
tts=sherpa_onnx.OfflineTts(cfg)
def say(text,out,speed=1.0):
    a=tts.generate(text,sid=0,speed=speed)
    sf.write(out,a.samples,samplerate=a.sample_rate)
    return len(a.samples)/a.sample_rate
if __name__=="__main__":
    print(say(sys.argv[1],sys.argv[2]))
