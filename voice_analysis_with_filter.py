import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import crepe
import librosa
import parselmouth
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, cosine
import warnings
import os
import platform
from scipy.signal import butter, lfilter

warnings.filterwarnings("ignore")

# === 中文字体支持 ===
def set_chinese_font():
    system = platform.system()
    if system == "Windows":
        zh_font = "SimHei"  # 黑体
    elif system == "Darwin":
        zh_font = "Heiti TC"
    else:
        zh_font = "SimHei"  # Linux下需先安装
    matplotlib.rcParams['font.family'] = zh_font
    matplotlib.rcParams['axes.unicode_minus'] = False

set_chinese_font()

# === 滤波器函数：带通滤波器用于降噪 ===
def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def denoise_audio(audio, sr, lowcut=150.0, highcut=600.0):
    b, a = butter_bandpass(lowcut, highcut, sr, order=6)
    return lfilter(b, a, audio)

# === Step 1: 加载并降噪音频 ===
def load_audio(path):
    audio, sr = librosa.load(path, sr=16000, mono=True)
    audio = denoise_audio(audio, sr)
    return audio, sr

# === Step 2: 使用 CREPE 提取音高 ===
def extract_pitch(audio, sr, threshold=0.6):
    time, freq, conf, _ = crepe.predict(audio, sr, viterbi=True, step_size=10)
    mask = conf > threshold
    return time[mask], freq[mask], conf[mask]

# === Step 3: 使用 Parselmouth 提取共振峰（F1 和 F2）===
def extract_formants(audio_path, step=0.01):
    snd = parselmouth.Sound(audio_path)
    formant = snd.to_formant_burg()
    f1_list, f2_list = [], []
    for t in np.arange(0, snd.duration, step):
        f1 = formant.get_value_at_time(1, t)
        f2 = formant.get_value_at_time(2, t)
        if f1 and f2 and not np.isnan(f1) and not np.isnan(f2):
            f1_list.append(f1)
            f2_list.append(f2)
    return np.array(f1_list), np.array(f2_list)

# === 新增：提取MIDI序列 ===
def extract_midi_features(time, freq, conf, midi_threshold=0.6):
    """从音高序列提取MIDI特征"""
    # 转换为MIDI音符编号 (0-127)
    midi_notes = librosa.hz_to_midi(freq)
    
    # 应用置信度阈值过滤
    mask = conf > midi_threshold
    filtered_time = time[mask]
    filtered_midi = midi_notes[mask]
    
    # 创建时间-MIDI对序列
    return np.array([(t, n) for t, n in zip(filtered_time, filtered_midi)])

# === Step 4: DTW 对齐两个序列 ===
def align_sequences(seq1, seq2):
    seq1_2d = seq1.reshape(-1, 1) if seq1.ndim == 1 else seq1
    seq2_2d = seq2.reshape(-1, 1) if seq2.ndim == 1 else seq2
    mask1 = ~np.any(np.isnan(seq1_2d), axis=1)
    mask2 = ~np.any(np.isnan(seq2_2d), axis=1)
    seq1_clean = seq1_2d[mask1]
    seq2_clean = seq2_2d[mask2]
    _, path = fastdtw(seq1_clean, seq2_clean, dist=euclidean)
    aligned_1, aligned_2 = [], []
    for i, j in path:
        aligned_1.append(seq1_clean[i])
        aligned_2.append(seq2_clean[j])
    return np.array(aligned_1), np.array(aligned_2)

# === 新增：计算MIDI序列相似度 ===
def calculate_midi_similarity(midi1, midi2, time_window=0.5):
    """计算两个MIDI序列的相似度（使用DTW对齐）"""
    # 如果有一个序列为空，返回0相似度
    if len(midi1) == 0 or len(midi2) == 0:
        return 0.0
    
    # 创建时间网格
    max_time = max(midi1[-1][0], midi2[-1][0])
    time_bins = np.linspace(0, max_time, int(max_time/time_window)+1)
    
    # 创建直方图矩阵
    def create_midi_hist(midi_notes):
        hist = np.zeros((len(time_bins)-1, 128))
        for (t, note) in midi_notes:
            bin_idx = np.searchsorted(time_bins, t) - 1
            if 0 <= bin_idx < len(hist):
                pitch = int(np.clip(note, 0, 127))
                hist[bin_idx, pitch] += 1
        return hist
    
    # 生成直方图
    hist1 = create_midi_hist(midi1)
    hist2 = create_midi_hist(midi2)
    
    # 使用DTW对齐直方图序列
    distance, path = fastdtw(hist1, hist2, dist=euclidean)
    
    # 计算最大可能距离（经验值）
    max_dist = np.sqrt(len(hist1)*len(hist2)) * 5
    similarity = max(0, 100 - (distance / max_dist * 100))
    return round(similarity, 2)

# === Step 5: 绘图展示对比结果 ===
def plot_results(pro_time, pro_pitch, user_time, user_pitch, deviation_cents,
                 aligned_f1_diff, aligned_f2_diff, midi_sim, save_path="comparison.png"):
    plt.figure(figsize=(14, 16))
    
    # 0: 显示整体相似度
    plt.subplot(5, 1, 1)
    plt.axis('off')
    plt.text(0.5, 0.5, 
             f"人声综合分析结果\nMIDI旋律相似度: {midi_sim}%", 
             ha='center', va='center', fontsize=14,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='blue', boxstyle='round,pad=0.5'))
    
    # 1: Pitch 曲线
    plt.subplot(5, 1, 2)
    plt.plot(pro_time, pro_pitch, label="专业歌手", color='blue')
    plt.plot(user_time, user_pitch, label="用户", color='red', alpha=0.6)
    plt.xlabel("时间 (秒)")
    plt.ylabel("音高 (Hz)")
    plt.title("音高曲线对比")
    plt.legend()
    plt.grid(True)

    # 2: 音高偏差（单位：音分 cents）
    plt.subplot(5, 1, 3)
    plt.plot(deviation_cents, color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.fill_between(range(len(deviation_cents)), -20, 20, color='green', alpha=0.2, label="±20音分 (优秀)")
    plt.fill_between(range(len(deviation_cents)), -50, 50, color='yellow', alpha=0.1, label="±50音分 (良好)")
    plt.title("音高偏差 (单位：音分 cents)")
    plt.xlabel("对齐帧")
    plt.ylabel("偏差")
    plt.legend()
    plt.grid(True)

    # 3: 共振峰差异
    plt.subplot(5, 1, 4)
    plt.plot(aligned_f1_diff, label='F1 差异', color='orange')
    plt.plot(aligned_f2_diff, label='F2 差异', color='green')
    plt.axhline(0, color='black', linestyle='--')
    plt.title("共振峰差异 (用户 - 专业)")
    plt.xlabel("对齐帧")
    plt.ylabel("差异 (Hz)")
    plt.legend()
    plt.grid(True)
    
    # 4: MIDI音符可视化
    plt.subplot(5, 1, 5)
    # 显示MIDI相似度结果
    plt.text(0.5, 0.8, f"MIDI旋律相似度: {midi_sim}%", 
             ha='center', va='center', fontsize=12)
    # 添加MIDI说明
    plt.text(0.5, 0.6, "（MIDI相似度反映旋律轮廓的相似性）", 
             ha='center', va='center', fontsize=10, color='gray')
    # 关闭坐标轴
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✅ 分析图已保存为: {save_path}")

# === Step 6: 主函数 ===
def analyze(pro_path, user_path):
    print("🎤 加载音频中...")
    pro_audio, pro_sr = load_audio(pro_path)
    user_audio, user_sr = load_audio(user_path)

    print("🎼 提取音高中...")
    pro_time, pro_pitch, pro_conf = extract_pitch(pro_audio, pro_sr)
    user_time, user_pitch, user_conf = extract_pitch(user_audio, user_sr)
    
    print("🎹 提取MIDI特征...")
    pro_midi = extract_midi_features(pro_time, pro_pitch, pro_conf)
    user_midi = extract_midi_features(user_time, user_pitch, user_conf)
    
    print("🔢 计算MIDI相似度...")
    midi_sim = calculate_midi_similarity(pro_midi, user_midi)
    print(f"  MIDI旋律相似度: {midi_sim}%")

    print("⏱ 正在对齐音高...")
    aligned_pro_pitch, aligned_user_pitch = align_sequences(pro_pitch, user_pitch)
    deviation_cents = 1200 * np.log2(aligned_user_pitch / aligned_pro_pitch + 1e-6)

    print("🔍 提取共振峰...")
    pro_f1, pro_f2 = extract_formants(pro_path)
    user_f1, user_f2 = extract_formants(user_path)

    print("⏱ 正在对齐共振峰...")
    aligned_pro_formants, aligned_user_formants = align_sequences(
        np.stack((pro_f1, pro_f2), axis=1),
        np.stack((user_f1, user_f2), axis=1)
    )
    aligned_f1_diff = aligned_user_formants[:, 0] - aligned_pro_formants[:, 0]
    aligned_f2_diff = aligned_user_formants[:, 1] - aligned_pro_formants[:, 1]

    print("📊 绘图中...")
    plot_results(pro_time, pro_pitch, user_time, user_pitch, deviation_cents,
                 aligned_f1_diff, aligned_f2_diff, midi_sim)

# === Step 7: 调用入口 ===
if __name__ == "__main__":
    # 专业歌手音频和用户音频
    pro_path = "output\\100\\vocals.wav"
    user_path = "output\\80\\vocals.wav"
    
    # 检查文件是否存在
    if not os.path.exists(pro_path):
        raise FileNotFoundError(f"专业音频文件不存在: {pro_path}")
    if not os.path.exists(user_path):
        raise FileNotFoundError(f"用户音频文件不存在: {user_path}")
    
    analyze(pro_path, user_path)
