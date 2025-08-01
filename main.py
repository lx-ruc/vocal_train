import numpy as np
import matplotlib.pyplot as plt
import crepe
import librosa
from scipy.io import wavfile
from dtw import dtw
import pretty_midi
import os
import warnings
import matplotlib
import platform

# 设置中文支持
try:
    # 根据操作系统设置不同的中文字体
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用黑体
    elif system == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Linux系统
    
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    CHINESE_SUPPORT = True
except:
    CHINESE_SUPPORT = False
    print("警告: 中文字体设置失败，将使用英文显示")

warnings.filterwarnings('ignore')

class VocalCoach:
    def __init__(self, pro_audio_path, user_audio_path):
        """
        初始化声乐教练系统
        :param pro_audio_path: 专业歌手音频路径
        :param user_audio_path: 用户音频路径
        """
        self.pro_audio, self.pro_sr = self.load_audio(pro_audio_path)
        self.user_audio, self.user_sr = self.load_audio(user_audio_path)
        self.pro_freq = None
        self.user_freq = None
        self.pro_time = None
        self.user_time = None
        self.alignment = None
        self.metrics = {}
        self.tips = []

    @staticmethod
    def load_audio(path):
        """加载音频文件并统一为单声道16kHz"""
        audio, sr = librosa.load(path, sr=16000, mono=True)
        return audio, sr

    def extract_pitch(self, confidence_threshold=0.6):
        """使用CREPE提取音高"""
        # 提取专业歌手音高
        pro_time, pro_freq, pro_conf, _ = crepe.predict(
            self.pro_audio, self.pro_sr, viterbi=True, step_size=10)
        pro_mask = pro_conf > confidence_threshold
        self.pro_time = pro_time[pro_mask]
        self.pro_freq = pro_freq[pro_mask]
        
        # 提取用户音高
        user_time, user_freq, user_conf, _ = crepe.predict(
            self.user_audio, self.user_sr, viterbi=True, step_size=10)
        user_mask = user_conf > confidence_threshold
        self.user_time = user_time[user_mask]
        self.user_freq = user_freq[user_mask]
        
        return self.pro_time, self.pro_freq, self.user_time, self.user_freq

    def analyze_pitch_deviation(self):
        """分析音高偏差"""
        # 动态时间规整对齐
        self.alignment = dtw(self.pro_freq, self.user_freq, keep_internals=True)
        
        # 计算音分偏差 (1200音分 = 1个八度)
        deviation_cents = 1200 * np.log2(
            self.user_freq[self.alignment.index2] / 
            self.pro_freq[self.alignment.index1]
        )
        
        # 计算关键指标
        abs_deviation = np.abs(deviation_cents)
        self.metrics = {
            'avg_deviation': np.mean(abs_deviation),
            'max_deviation': np.max(abs_deviation),
            'under_pitch': np.mean(deviation_cents < -20),
            'over_pitch': np.mean(deviation_cents > 20),
            'stability': np.std(self.user_freq),
            'vibrato': self.calculate_vibrato(self.user_freq)
        }
        
        # 返回对齐后的专业歌手时间点
        aligned_pro_time = self.pro_time[self.alignment.index1]
        return aligned_pro_time, deviation_cents

    @staticmethod
    def calculate_vibrato(freq, window_size=30, threshold=3):
        """计算颤音深度"""
        if len(freq) < window_size * 2:
            return 0
        
        # 使用滑动窗口计算频率变化
        variations = []
        for i in range(len(freq) - window_size):
            segment = freq[i:i+window_size]
            if np.max(segment) > 0 and np.min(segment) > 0:
                # 计算半音变化 (100音分 = 1半音)
                semitones = 12 * np.log2(np.max(segment)/np.min(segment))
                variations.append(semitones)
        
        # 过滤微小波动
        significant = [v for v in variations if v > threshold/100]
        return np.mean(significant) * 100 if significant else 0

    def generate_tips(self):
        """生成发音技巧提示"""
        self.tips = []
        metrics = self.metrics
        
        # 音准偏差提示
        if metrics['avg_deviation'] > 50:
            self.tips.append("⚠️ 整体音准偏差较大（平均 {:.1f} 音分）".format(metrics['avg_deviation']))
            if metrics['under_pitch'] > 0.6:
                self.tips.append("🔻 持续偏低问题：喉部肌肉紧张，尝试：")
                self.tips.append("  1. 降低喉位，想象'打哈欠'状态")
                self.tips.append("  2. 增强气息支撑，使用腹式呼吸")
            elif metrics['over_pitch'] > 0.6:
                self.tips.append("🔺 持续偏高问题：气息不足导致，建议：")
                self.tips.append("  1. 增大共鸣空间，提高软腭")
                self.tips.append("  2. 减小音量，避免挤压声带")
        
        # 稳定性提示
        if metrics['stability'] > 30:
            self.tips.append("🎯 音高稳定性问题（波动 {:.1f} 音分）：".format(metrics['stability']))
            self.tips.append("  1. 练习平稳长音（'a'音持续5秒）")
            self.tips.append("  2. 减小气息流量，保持均匀呼气")
        
        # 颤音提示
        if metrics['vibrato'] > 80:
            self.tips.append("🎶 颤音幅度过大（{:.1f} 音分）：".format(metrics['vibrato']))
            self.tips.append("  1. 控制腹部肌肉，减少波动")
            self.tips.append("  2. 练习直音稳定后再加颤音")
        elif 0 < metrics['vibrato'] < 30:
            self.tips.append("🎵 颤音幅度不足（{:.1f} 音分）：".format(metrics['vibrato']))
            self.tips.append("  1. 放松喉部，让声音自然波动")
            self.tips.append("  2. 练习'波浪式'气息控制")
        
        # 高音区专项提示
        high_notes = [f for f in self.user_freq if f > 500]  # G5以上
        if high_notes:
            high_dev = np.mean(np.abs(1200 * np.log2(np.array(high_notes)/500)))
            if high_dev > 60:
                self.tips.append("🚀 高音区问题（偏差 {:.1f} 音分）：".format(high_dev))
                self.tips.append("  1. 加强头腔共鸣，使用'ng'音练习")
                self.tips.append("  2. 减小发声点，想象声音从眉心发出")
        
        # 添加个性化鼓励
        if metrics['avg_deviation'] < 30:
            self.tips.append("🎉 优秀！音准控制接近专业水平")
        elif metrics['avg_deviation'] < 50:
            self.tips.append("👍 良好表现，继续提升稳定性")
        else:
            self.tips.append("💪 坚持练习，每天进步一点点！")
        
        return self.tips

    def visualize_comparison(self, aligned_pro_time, deviation_cents, output_path="comparison.png"):
        """可视化音高对比 - 支持中文/英文显示"""
        plt.figure(figsize=(14, 10))
        
        # 根据中文支持情况设置标签
        if CHINESE_SUPPORT:
            # 中文标签
            title1 = '音高曲线对比'
            title2 = '音高偏差分析'
            xlabel = '时间 (秒)'
            ylabel1 = '音高频率 (Hz)'
            ylabel2 = '偏差值 (音分)'
            legend1 = ['专业歌手', '用户']
            legend2 = ['优秀区 (±20音分)', '可接受区 (±50音分)']
            colorbar_label = '偏差绝对值 (音分)'
            
            # 图例说明框文本
            text1 = ("图例说明:\n"
                     "• 蓝色曲线: 专业歌手音高轨迹\n"
                     "• 红色曲线: 用户音高轨迹\n"
                     "• 纵坐标: 声音频率(Hz)，数值越高音高越高\n"
                     "• 横坐标: 时间(秒)，显示音频时长")
            
            text2 = ("图例说明:\n"
                     "• 每个点: 用户与专业歌手在该时刻的音高差异\n"
                     "• 颜色: 红色表示偏差大，蓝色表示偏差小\n"
                     "• 绿区: 优秀音准 (±20音分)\n"
                     "• 黄区: 可接受范围 (±50音分)\n"
                     "• 纵坐标: 正值为偏高，负值为偏低")
        else:
            # 英文标签
            title1 = 'Pitch Curve Comparison'
            title2 = 'Pitch Deviation Analysis'
            xlabel = 'Time (seconds)'
            ylabel1 = 'Pitch Frequency (Hz)'
            ylabel2 = 'Deviation (cents)'
            legend1 = ['Professional Singer', 'User']
            legend2 = ['Excellent (±20 cents)', 'Acceptable (±50 cents)']
            colorbar_label = 'Absolute Deviation (cents)'
            
            # 图例说明框文本
            text1 = ("Legend:\n"
                     "• Blue curve: Professional singer's pitch trajectory\n"
                     "• Red curve: User's pitch trajectory\n"
                     "• Y-axis: Sound frequency (Hz), higher values = higher pitch\n"
                     "• X-axis: Time (seconds), showing audio duration")
            
            text2 = ("Legend:\n"
                     "• Each point: Pitch difference at that moment\n"
                     "• Color: Red = large deviation, Blue = small deviation\n"
                     "• Green zone: Excellent pitch accuracy (±20 cents)\n"
                     "• Yellow zone: Acceptable range (±50 cents)\n"
                     "• Y-axis: Positive = sharp, Negative = flat")

        # 音高曲线对比
        plt.subplot(2, 1, 1)
        plt.plot(self.pro_time, self.pro_freq, 'b-', alpha=0.7, label=legend1[0])
        plt.plot(self.user_time, self.user_freq, 'r-', alpha=0.5, label=legend1[1])
        
        # 添加详细图例说明
        plt.title(title1, fontsize=14, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel1, fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加图例说明框
        plt.figtext(0.15, 0.01, text1, 
                    bbox=dict(facecolor='white', alpha=0.8), 
                    fontsize=9)
        
        # 偏差热力图 - 使用对齐后的时间点
        plt.subplot(2, 1, 2)
        plt.scatter(aligned_pro_time, deviation_cents, 
                    c=np.abs(deviation_cents), cmap='coolwarm', 
                    alpha=0.6, marker='.', s=20)
        plt.colorbar(label=colorbar_label, pad=0.01)
        plt.axhline(0, color='black', linestyle='--')
        plt.fill_between(aligned_pro_time, -20, 20, 
                         color='green', alpha=0.15, label=legend2[0])
        plt.fill_between(aligned_pro_time, -50, 50, 
                         color='yellow', alpha=0.1, label=legend2[1])
        
        # 添加详细图例说明
        plt.title(title2, fontsize=14, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel2, fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加图例说明框
        plt.figtext(0.15, 0.45, text2, 
                    bbox=dict(facecolor='white', alpha=0.8), 
                    fontsize=9)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 为底部文本留空间
        plt.savefig(output_path, dpi=150)
        print(f"可视化结果已保存至: {output_path}")
        return output_path

    def generate_report(self, output_dir="output"):
        """生成完整分析报告"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 分析音高
        pro_time, pro_freq, user_time, user_freq = self.extract_pitch()
        
        # 分析音高偏差并获取对齐后的时间点
        aligned_pro_time, deviation_cents = self.analyze_pitch_deviation()
        
        tips = self.generate_tips()
        img_path = self.visualize_comparison(aligned_pro_time, deviation_cents, 
                    os.path.join(output_dir, "pitch_comparison.png"))
        
        # 计算音频时长
        pro_duration = f"{pro_time[-1]:.1f}" if len(pro_time) > 0 else "N/A"
        user_duration = f"{user_time[-1]:.1f}" if len(user_time) > 0 else "N/A"
        
        # 根据中文支持设置报告标题
        if CHINESE_SUPPORT:
            report_title = "声乐训练分析报告"
            headers = [
                f"专业歌手音频: {pro_duration}秒 | 音高范围: {np.min(pro_freq):.1f}-{np.max(pro_freq):.1f} Hz",
                f"用户音频: {user_duration}秒 | 音高范围: {np.min(user_freq):.1f}-{np.max(user_freq):.1f} Hz",
                f"平均偏差: {self.metrics['avg_deviation']:.1f} 音分",
                f"最大偏差: {self.metrics['max_deviation']:.1f} 音分",
                f"音高稳定性: {self.metrics['stability']:.1f} (标准差)",
                f"颤音深度: {self.metrics['vibrato']:.1f} 音分",
                "发音技巧建议:"
            ]
        else:
            report_title = "Vocal Training Analysis Report"
            headers = [
                f"Professional Audio: {pro_duration} sec | Pitch Range: {np.min(pro_freq):.1f}-{np.max(pro_freq):.1f} Hz",
                f"User Audio: {user_duration} sec | Pitch Range: {np.min(user_freq):.1f}-{np.max(user_freq):.1f} Hz",
                f"Average Deviation: {self.metrics['avg_deviation']:.1f} cents",
                f"Max Deviation: {self.metrics['max_deviation']:.1f} cents",
                f"Pitch Stability: {self.metrics['stability']:.1f} (std dev)",
                f"Vibrato Depth: {self.metrics['vibrato']:.1f} cents",
                "Vocal Technique Suggestions:"
            ]
        
        # 生成文本报告
        report = [
            "="*60,
            report_title,
            "="*60,
            headers[0],
            headers[1],
            "-"*60,
            headers[2],
            headers[3],
            headers[4],
            headers[5],
            "-"*60,
            headers[6]
        ]
        report.extend(self.tips)
        report.append("="*60)
        
        # 保存报告
        report_path = os.path.join(output_dir, "vocal_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
        
        print(f"分析报告已保存至: {report_path}")
        return report_path, img_path

if __name__ == "__main__":
    # 使用示例
    pro_audio = "beyond_origin.wav"  # 专业歌手音频
    user_audio = "beyond_db.wav"      # 用户音频
    
    # 根据中文支持设置界面语言
    if CHINESE_SUPPORT:
        print("="*60)
        print("声乐训练分析系统")
        print("="*60)
        print("正在分析音频...")
    else:
        print("="*60)
        print("Vocal Training Analysis System")
        print("="*60)
        print("Analyzing audio...")
    
    coach = VocalCoach(pro_audio, user_audio)
    report_path, img_path = coach.generate_report()
    
    # 打印报告摘要
    if CHINESE_SUPPORT:
        print("\n分析摘要:")
    else:
        print("\nAnalysis Summary:")
    
    if len(coach.pro_time) > 0:
        if CHINESE_SUPPORT:
            print(f"专业歌手音频时长: {coach.pro_time[-1]:.1f}秒")
        else:
            print(f"Professional Audio Duration: {coach.pro_time[-1]:.1f} sec")
    else:
        if CHINESE_SUPPORT:
            print("专业歌手音频时长: N/A")
        else:
            print("Professional Audio Duration: N/A")
            
    if len(coach.user_time) > 0:
        if CHINESE_SUPPORT:
            print(f"用户音频时长: {coach.user_time[-1]:.1f}秒")
        else:
            print(f"User Audio Duration: {coach.user_time[-1]:.1f} sec")
    else:
        if CHINESE_SUPPORT:
            print("用户音频时长: N/A")
        else:
            print("User Audio Duration: N/A")
            
    if CHINESE_SUPPORT:
        print(f"平均音准偏差: {coach.metrics['avg_deviation']:.1f} 音分")
        print(f"最大偏差: {coach.metrics['max_deviation']:.1f} 音分")
        print("\n主要建议:")
    else:
        print(f"Average Pitch Deviation: {coach.metrics['avg_deviation']:.1f} cents")
        print(f"Max Deviation: {coach.metrics['max_deviation']:.1f} cents")
        print("\nKey Suggestions:")
    
    for tip in coach.tips[:3]:
        print(f" - {tip}")
    
    if CHINESE_SUPPORT:
        print("\n请查看output文件夹获取完整报告和图表")
    else:
        print("\nPlease check the 'output' folder for full report and charts")
