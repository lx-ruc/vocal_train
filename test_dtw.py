import librosa
import crepe
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def extract_pitch_crepe(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    _, pitch, confidence, _ = crepe.predict(audio, sr, viterbi=True)
    return pitch.flatten()

def plot_pitch_difference(ref_pitch, test_pitch):
    ref_2d = ref_pitch.reshape(-1, 1)
    test_2d = test_pitch.reshape(-1, 1)
    
    _, path = fastdtw(ref_2d, test_2d, dist=euclidean)
    
    pitch_diffs = []
    for i, j in path:
        r, t = ref_pitch[i], test_pitch[j]
        if r > 0 and t > 0:
            pitch_diffs.append(t - r)
        else:
            pitch_diffs.append(0)
    
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(pitch_diffs)), pitch_diffs, 'bo-', label="音高差 (用户 - 专业)")
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("每帧音高差异图")
    plt.xlabel("对齐帧编号")
    plt.ylabel("音高差异 (Hz)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 替换成你自己的音频文件路径
    ref_audio_path = "beyond_origin.m4a"
    test_audio_path = "beyond_db.m4a"
    
    ref_pitch = extract_pitch_crepe(ref_audio_path)
    test_pitch = extract_pitch_crepe(test_audio_path)
    
    plot_pitch_difference(ref_pitch, test_pitch)
