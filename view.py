import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import os

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载CSV数据
file_path = 'beyond.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"找不到文件: {file_path}")

try:
    df = pd.read_csv(file_path)
    print(f"成功加载文件: {file_path}")
    print(f"数据维度: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 标准化列名（不区分大小写）
    df.columns = df.columns.str.lower()
    col_map = {
        'time': ['time', 'timestamp', 't'],
        'frequency': ['frequency', 'freq', 'f0', 'pitch'],
        'confidence': ['confidence', 'conf', 'prob']
    }
    
    # 尝试匹配列名
    matched_cols = {}
    for standard, variants in col_map.items():
        for col in df.columns:
            if col in variants:
                matched_cols[standard] = col
                break
    
    # 检查必要列是否存在
    if not all(key in matched_cols for key in ['time', 'frequency', 'confidence']):
        missing = [key for key in ['time', 'frequency', 'confidence'] if key not in matched_cols]
        raise ValueError(f"缺少必要的列: {missing}。请确保CSV包含时间、频率和置信度列")
    
    # 重命名列
    df = df.rename(columns={
        matched_cols['time']: 'time',
        matched_cols['frequency']: 'frequency',
        matched_cols['confidence']: 'confidence'
    })
    
    # 显示数据预览
    print("\n数据预览:")
    print(df.head(3))
    
except Exception as e:
    print(f"加载文件出错: {str(e)}")
    exit(1)

# 2. 数据预处理
print("\n数据处理中...")

# 转换频率为MIDI音符号
df['midi_note'] = 12 * np.log2(df['frequency'] / 440.0) + 69
df['midi_note'] = df['midi_note'].replace([np.inf, -np.inf], np.nan)

# 过滤无效频率
initial_count = len(df)
df = df[(df['frequency'] > 50) & (df['frequency'] < 2000)].copy()
filtered_count = initial_count - len(df)
print(f"过滤了 {filtered_count} 个无效频率点 ({filtered_count/initial_count*100:.1f}%)")

# 添加时间分钟格式
df['time_min'] = df['time'] / 60

# 音符名称映射
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def midi_to_note(midi_val):
    if pd.isna(midi_val) or midi_val < 0:
        return ""
    note_index = int(round(midi_val)) % 12
    octave = int(round(midi_val)) // 12 - 1
    return f"{note_names[note_index]}{octave}"

df['note'] = df['midi_note'].apply(midi_to_note)

# 3. 确定音域范围
print("\n分析音域范围...")
valid_df = df[df['confidence'] > 0.7]
if len(valid_df) > 0:
    min_note = np.floor(valid_df['midi_note'].min())
    max_note = np.ceil(valid_df['midi_note'].max())
    note_range = max_note - min_note
    print(f"检测到音域范围: {min_note:.0f}-{max_note:.0f} (MIDI), 跨度: {note_range:.1f} 半音")
    
    # 扩展音域范围确保包含所有数据
    y_min = max(40, min_note - 5)  # 最低不低于40 (E2)
    y_max = min(100, max_note + 5)  # 最高不超过100 (E7)
    y_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + 1, 2)
else:
    print("警告: 未找到高置信度数据，使用默认音域范围")
    y_min, y_max = 48, 84  # C3-C6
    y_ticks = np.arange(48, 85, 4)

# 4. 创建专业级可视化
print("\n创建可视化图表...")
plt.figure(figsize=(16, 14))
plt.suptitle(f'音高分析报告: {os.path.basename(file_path)}', fontsize=18, fontweight='bold')
gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1], width_ratios=[3, 1])

# 主图：音高曲线
ax1 = plt.subplot(gs[0, :])
plt.sca(ax1)

# 绘制置信度背景
time_min = df['time_min'].min()
time_max = df['time_min'].max()
time_range = time_max - time_min

# 创建时间网格用于背景着色
x_grid = np.linspace(time_min, time_max, 100)
confidence_interp = np.interp(x_grid, df['time_min'], df['confidence'])
colors = plt.cm.coolwarm(confidence_interp)
for i in range(len(x_grid)-1):
    ax1.axvspan(x_grid[i], x_grid[i+1], color=colors[i], alpha=0.1)

# 绘制音高曲线（按置信度分组）
high_conf = df[df['confidence'] >= 0.7]
low_conf = df[df['confidence'] < 0.7]

if not high_conf.empty:
    ax1.scatter(high_conf['time_min'], high_conf['midi_note'], 
                c='darkblue', s=12, alpha=0.7, label='高置信度 (≥0.7)')
if not low_conf.empty:
    ax1.scatter(low_conf['time_min'], low_conf['midi_note'], 
                c='lightgray', s=8, alpha=0.4, label='低置信度 (<0.7)')

# 添加参考线
for y in y_ticks:
    if y % 12 == 0:  # C音加粗显示
        ax1.axhline(y=y, color='red', linestyle='-', alpha=0.3, lw=1.5)
    else:
        ax1.axhline(y=y, color='gray', linestyle='--', alpha=0.2, lw=0.8)

# 设置Y轴为音符名
ax1.set_yticks(y_ticks)
ax1.set_yticklabels([midi_to_note(y) for y in y_ticks])
ax1.set_ylim(y_min, y_max)
ax1.set_xlim(time_min, time_max)

ax1.set_ylabel('音高 (音符)', fontsize=12)
ax1.set_xlabel('时间 (分钟)', fontsize=12)
ax1.set_title('音高曲线 - 背景色表示置信度 (蓝色=高, 红色=低)', fontsize=14)
ax1.grid(True, alpha=0.2)
ax1.legend(loc='upper right')

# 5. 置信度分析
ax2 = plt.subplot(gs[1, 0])
plt.sca(ax2)

# 置信度分布
sns.histplot(df['confidence'], bins=30, kde=True, color='teal', ax=ax2)
ax2.axvline(x=0.7, color='r', linestyle='--', label='推荐阈值 (0.7)')
ax2.set_title('置信度分布', fontsize=13)
ax2.set_xlabel('置信度', fontsize=11)
ax2.set_ylabel('数据点数量', fontsize=11)
ax2.legend()

# 添加统计信息
conf_mean = df['confidence'].mean()
conf_median = df['confidence'].median()
ax2.text(0.05, 0.9, f'均值: {conf_mean:.2f}\n中位数: {conf_median:.2f}', 
         transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))

# 6. 音高分布分析
ax3 = plt.subplot(gs[1, 1])
plt.sca(ax3)

if not valid_df.empty and len(valid_df) > 10:
    # 音高直方图
    sns.histplot(valid_df['midi_note'], bins=24, kde=True, color='purple', ax=ax3)
    
    # 添加垂直线表示平均音高
    mean_pitch = valid_df['midi_note'].mean()
    ax3.axvline(x=mean_pitch, color='r', linestyle='--', label=f'平均音高 ({mean_pitch:.1f})')
    
    ax3.set_title('有效音高分布 (置信度>0.7)', fontsize=13)
    ax3.set_xlabel('MIDI音符值', fontsize=11)
    ax3.set_ylabel('出现频率', fontsize=11)
    ax3.legend()
else:
    ax3.text(0.5, 0.5, '无足够高置信度数据', 
             ha='center', va='center', fontsize=12, 
             bbox=dict(facecolor='lightyellow', alpha=0.7))
    ax3.set_title('有效音高分布 (无足够数据)', fontsize=13)

# 7. 时间特性分析
ax4 = plt.subplot(gs[2, 0])
plt.sca(ax4)

if not valid_df.empty and len(valid_df) > 100:
    # 计算每30秒窗口内的平均音高
    valid_df['time_window'] = (valid_df['time'] // 30) * 30
    time_series = valid_df.groupby('time_window')['midi_note'].mean().reset_index()
    
    # 转换为分钟
    time_series['time_min'] = time_series['time_window'] / 60
    
    ax4.plot(time_series['time_min'], time_series['midi_note'], 'o-', color='green', linewidth=2)
    ax4.fill_between(time_series['time_min'], 
                     time_series['midi_note'] - 1, 
                     time_series['midi_note'] + 1, 
                     color='green', alpha=0.1)
    
    ax4.set_title('平均音高随时间变化 (30秒窗口)', fontsize=13)
    ax4.set_xlabel('时间 (分钟)', fontsize=11)
    ax4.set_ylabel('平均MIDI音符值', fontsize=11)
    ax4.grid(alpha=0.2)
else:
    ax4.text(0.5, 0.5, '无足够高置信度数据', 
             ha='center', va='center', fontsize=12, 
             bbox=dict(facecolor='lightyellow', alpha=0.7))
    ax4.set_title('平均音高随时间变化 (无足够数据)', fontsize=13)

# 8. 统计信息面板
ax5 = plt.subplot(gs[2, 1])
plt.sca(ax5)
ax5.axis('off')

# 创建统计文本
stats_text = [
    f"文件: {os.path.basename(file_path)}",
    f"总时长: {df['time'].max():.1f} 秒 ({df['time'].max()/60:.1f} 分钟)",
    f"数据点数: {len(df):,}",
    f"有效音高比例: {len(valid_df)/len(df)*100:.1f}% (置信度>0.7)"
]

if not valid_df.empty:
    stats_text.extend([
        "",
        "--- 高置信度数据统计 ---",
        f"最高音: {midi_to_note(valid_df['midi_note'].max())} ({valid_df['midi_note'].max():.1f})",
        f"最低音: {midi_to_note(valid_df['midi_note'].min())} ({valid_df['midi_note'].min():.1f})",
        f"平均音高: {valid_df['midi_note'].mean():.1f}",
        f"音高标准差: {valid_df['midi_note'].std():.2f} 半音",
        f"最稳定音符: {valid_df['note'].value_counts().idxmax()}"
    ])

# 添加统计文本
for i, text in enumerate(stats_text):
    ax5.text(0.05, 0.95 - i*0.06, text, fontsize=11, 
             transform=ax5.transAxes, 
             bbox=dict(facecolor='lightblue', alpha=0.2) if i > 3 else None)

# 添加数据质量评估
quality_score = len(valid_df) / len(df) * conf_mean * 100
quality_text = f"数据质量评分: {quality_score:.1f}/100\n"
if quality_score > 70:
    quality_text += "质量优秀 ✓"
elif quality_score > 50:
    quality_text += "质量良好 ~"
else:
    quality_text += "质量较差 ✗"

ax5.text(0.05, 0.15, quality_text, fontsize=12, 
         transform=ax5.transAxes, 
         bbox=dict(facecolor='lightgreen' if quality_score > 70 else 
                   'lightyellow' if quality_score > 50 else 
                   'lightcoral', alpha=0.5))

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.93, hspace=0.35, wspace=0.25)

# 保存和显示
output_file = f"{os.path.splitext(file_path)[0]}_analysis.png"
plt.savefig(output_file, bbox_inches='tight', dpi=150)
print(f"\n分析完成! 结果已保存至: {output_file}")
plt.show()