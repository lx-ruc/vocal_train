# -*- coding: utf-8 -*-
"""
Vocal Similarity to Reference (Pitch/Volume + Rhythm from HMM)
- 节奏部分使用论文原始的三类 HMM 方法
- 仅计算文件夹中每个音频与标准人声的相似度
- 输出CSV和总分柱状图

用法：
    python vocal_to_ref_hmm.py --folder D:/voice/vocals --ref standard.wav
"""
import os
import argparse
import warnings
import numpy as np
import librosa
import matplotlib.pyplot as plt
from hmmlearn import hmm

warnings.filterwarnings("ignore", category=UserWarning)

# =====================
# 工具函数
# =====================
def hz_to_midi_safe(f0_hz: np.ndarray):
    f0_midi = librosa.hz_to_midi(f0_hz)
    valid = ~np.isnan(f0_midi)
    if valid.sum() >= 2:
        out = f0_midi.copy()
        idx = np.arange(len(f0_midi))
        out[~valid] = np.interp(idx[~valid], idx[valid], f0_midi[valid])
        return out, valid
    mean_val = np.nanmean(f0_midi)
    if not np.isfinite(mean_val):
        mean_val = 0.0
    return np.nan_to_num(f0_midi, nan=mean_val), valid

def rolling_median(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k < 2 or len(x) < 3:
        return x
    if k % 2 == 0:
        k += 1
    half = k // 2
    n = len(x)
    y = np.empty_like(x)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        y[i] = np.median(x[lo:hi])
    return y

def seq_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - np.nanmean(x)) / (np.nanstd(x) + eps)

def _cost_matrix_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    return np.abs(x[:, None] - y[None, :])

def dtw_similarity(x: np.ndarray, y: np.ndarray, gamma: float = 0.02):
    C = _cost_matrix_1d(x, y)
    D, wp = librosa.sequence.dtw(C=C)
    dist = float(D[-1, -1]) / max(1, len(wp))
    sim = 100.0 * float(np.exp(-gamma * dist))
    return sim, dist

# =====================
# 节奏 HMM 模块（论文方法）
# =====================
def extract_strength_sequence(y, sr, hop_length=256, n_bins=60):
    """用CQT计算强度向量，每帧取总强度作为特征"""
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins))
    strength = C.sum(axis=0)
    return seq_zscore(strength).reshape(-1, 1)  # HMM 要求二维输入

def shift_sequence(seq, shift_frames):
    """循环平移序列，模拟提前或延迟唱"""
    if shift_frames == 0:
        return seq
    return np.roll(seq, shift_frames, axis=0)

def train_rhythm_hmms(ref_seq, n_states=7, n_mix=4):
    """
    训练三类HMM：
    λ1: 同步（原始参考）
    λ2: 超前（向左移）
    λ3: 落后（向右移）
    """
    shift_frames = 5  # 模拟提前/延迟的帧数
    seq_sync = ref_seq
    seq_ahead = shift_sequence(ref_seq, -shift_frames)
    seq_late = shift_sequence(ref_seq, shift_frames)

    models = []
    for seq in [seq_sync, seq_ahead, seq_late]:
        model = hmm.GMMHMM(n_components=n_states, n_mix=n_mix, covariance_type="diag", n_iter=100)
        model.fit(seq)
        models.append(model)
    return models

def rhythm_similarity_hmm(models, ref_seq, test_seq, seg_len=20):
    """
    将测试段分片，用HMM判断每片同步类别
    节奏得分 = 100 * (1 - 非同步比例)
    """
    n_frames = min(len(ref_seq), len(test_seq))
    ref_seq = ref_seq[:n_frames]
    test_seq = test_seq[:n_frames]

    # 按论文思路，这里其实要对 ref_seq 和 test_seq 做对齐，但简化为直接比较
    non_sync_count = 0
    total_segs = 0
    for start in range(0, n_frames - seg_len + 1, seg_len):
        seg = test_seq[start:start+seg_len]
        # 计算三类模型的概率
        scores = [m.score(seg) for m in models]
        best_class = np.argmax(scores) + 1  # 1=同步, 2=超前, 3=落后
        if best_class != 1:
            non_sync_count += 1
        total_segs += 1
    if total_segs == 0:
        return 0.0
    return 100.0 * (1 - non_sync_count / total_segs)

# =====================
# 核心：相似度计算
# =====================
def vocal_similarity_with_hmm(
    ref_path, test_path, sr=22050,
    frame_length=2048, hop_length=256,
    pit_weight=0.5, vol_weight=0.2, rhy_weight=0.3
):
    # 读取
    y_ref, sr1 = librosa.load(ref_path, sr=sr, mono=True)
    y_test, sr2 = librosa.load(test_path, sr=sr, mono=True)
    if sr1 != sr2:
        raise ValueError("采样率不一致")
    sr = sr1

    # 音高
    f0_r, _, _ = librosa.pyin(y_ref, sr=sr, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), frame_length=frame_length, hop_length=hop_length)
    f0_t, _, _ = librosa.pyin(y_test, sr=sr, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), frame_length=frame_length, hop_length=hop_length)
    midi_r, _ = hz_to_midi_safe(f0_r)
    midi_t, _ = hz_to_midi_safe(f0_t)
    midi_r = rolling_median(midi_r, k=5)
    midi_t = rolling_median(midi_t, k=5)
    S_pit, _ = dtw_similarity(midi_r, midi_t, gamma=0.05)

    # 音量
    rms_r = librosa.feature.rms(y=y_ref, frame_length=frame_length, hop_length=hop_length).ravel()
    rms_t = librosa.feature.rms(y=y_test, frame_length=frame_length, hop_length=hop_length).ravel()
    S_vol, _ = dtw_similarity(seq_zscore(rms_r), seq_zscore(rms_t), gamma=0.02)

    # 节奏（HMM）
    ref_seq = extract_strength_sequence(y_ref, sr, hop_length)
    test_seq = extract_strength_sequence(y_test, sr, hop_length)
    hmms = train_rhythm_hmms(ref_seq)
    S_rhy = rhythm_similarity_hmm(hmms, ref_seq, test_seq)

    total = np.clip(pit_weight * S_pit + vol_weight * S_vol + rhy_weight * S_rhy, 0.0, 100.0)
    return {
        "S_total": float(total),
        "S_pitch": float(S_pit),
        "S_volume": float(S_vol),
        "S_rhythm": float(S_rhy)
    }

# =====================
# 主程序
# =====================
def main():
    parser = argparse.ArgumentParser(description="Vocal similarity to reference (HMM rhythm).")
    parser.add_argument("--folder", type=str, required=True, help="包含人声文件的文件夹")
    parser.add_argument("--ref", type=str, required=True, help="标准人声文件名（在folder中）")
    parser.add_argument("--sr", type=int, default=22050, help="采样率")
    parser.add_argument("--csv", type=str, default="similarity_to_ref.csv", help="结果CSV保存路径")
    parser.add_argument("--png", type=str, default="similarity_bar.png", help="柱状图保存路径")
    args = parser.parse_args()

    folder = args.folder
    ref_name = args.ref
    ref_path = os.path.join(folder, ref_name)
    if not os.path.isfile(ref_path):
        raise FileNotFoundError(f"标准人声文件不存在: {ref_path}")

    exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts) and f != ref_name]
    files.sort()
    if not files:
        raise ValueError("文件夹中没有待评人声文件")

    results = []
    for f in files:
        path = os.path.join(folder, f)
        res = vocal_similarity_with_hmm(ref_path, path, sr=args.sr)
        results.append((f, res["S_total"], res["S_pitch"], res["S_volume"], res["S_rhythm"]))
        print(f"{f}: total={res['S_total']:.2f}, pitch={res['S_pitch']:.2f}, vol={res['S_volume']:.2f}, rhy={res['S_rhythm']:.2f}")

    # 保存CSV
    with open(args.csv, "w", encoding="utf-8") as f:
        f.write("file,S_total,S_pitch,S_volume,S_rhythm\n")
        for row in results:
            f.write(f"{row[0]},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f}\n")
    print(f"[Saved] CSV -> {args.csv}")

    # 柱状图
    names = [r[0] for r in results]
    totals = [r[1] for r in results]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, totals, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 100)
    plt.ylabel("Similarity (0-100)")
    plt.title(f"Similarity to {ref_name}")
    for bar, val in zip(bars, totals):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(args.png, dpi=300)
    plt.close()
    print(f"[Saved] Bar chart -> {args.png}")

if __name__ == "__main__":
    main()
