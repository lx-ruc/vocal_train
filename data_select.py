import crepe
import librosa
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载音频
audio_pro, sr = librosa.load("pro_singer.wav", sr=16000)
audio_amateur, _ = librosa.load("amateur.wav", sr=16000)

# 2. 音高提取（CREPE）
_, f0_pro, conf_pro, _ = crepe.predict(audio_pro, sr, viterbi=True)
_, f0_amateur, conf_amateur, _ = crepe.predict(audio_amateur, sr, viterbi=True)

# 3. 时间对齐（DTW）
mfcc_pro = librosa.feature.mfcc(y=audio_pro, sr=sr, n_mfcc=13)
mfcc_amateur = librosa.feature.mfcc(y=audio_amateur, sr=sr, n_mfcc=13)
alignment = librosa.sequence.dtw(mfcc_pro, mfcc_amateur)

# 4. 计算音分差异
f0_pro_aligned = f0_pro[alignment[0]]  # 按DTW路径对齐专业歌手音高
f0_amateur_aligned = f0_amateur[alignment[1]]
diff_cents = 1200 * np.log2(f0_amateur_aligned / f0_pro_aligned)

# 5. 生成可视化报告
plt.figure(figsize=(15, 5))
plt.plot(f0_pro_aligned, label="Professional", lw=2)
plt.plot(f0_amateur_aligned, label="Amateur", alpha=0.8)
plt.fill_between(range(len(diff_cents)), -50, 50, color="yellow", alpha=0.2)  # 容差区间
plt.legend(); plt.title("Pitch Comparison"); plt.ylabel("Frequency (Hz)")
plt.savefig("pitch_comparison.png")