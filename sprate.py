from spleeter.separator import Separator
import os

def batch_separate_audio(input_dir, output_dir):
    """ 批量处理WAV文件（兼容大小写） """
    os.makedirs(output_dir, exist_ok=True)
    
    separator = Separator('spleeter:2stems')
    processed_files = 0

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.wav'):  # 兼容大小写
            input_path = os.path.join(input_dir, filename)
            print(f"正在处理: {filename}")
            
            try:
                separator.separate_to_file(input_path, output_dir)
                processed_files += 1
            except Exception as e:
                print(f"处理失败: {filename} - {str(e)}")

    print(f"\n完成! 共处理 {processed_files} 个文件")
    print(f"输出目录: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    batch_separate_audio(
        os.path.join(current_dir, "alone"),
        os.path.join(current_dir, "output_alone")
    )
