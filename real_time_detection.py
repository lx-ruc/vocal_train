import pyaudio
import numpy as np

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("实时音高检测中... (按Ctrl+C停止)")

try:
    while True:
        # 读取音频块
        data = stream.read(CHUNK)
        audio = np.frombuffer(data, dtype=np.float32)
        
        # 使用crepe处理
        _, freq, conf, _ = crepe.predict(audio, RATE, step_size=10, viterbi=True)
        
        # 取最高置信度的音高
        if np.max(conf) > 0.8:
            main_pitch = freq[np.argmax(conf)]
            print(f"当前音高: {main_pitch:.1f} Hz | 音符: {midi_to_note(12*np.log2(main_pitch/440)+69)}")
        else:
            print("未检测到有效音高")
            
except KeyboardInterrupt:
    stream.stop_stream()
    stream.close()
    p.terminate()