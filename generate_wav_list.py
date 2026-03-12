import os
import argparse
from pathlib import Path
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_wav_lists(dataset_root):
    """
    为每个Family文件夹生成train_wav_list.txt和test_wav_list.txt
    
    Args:
        dataset_root: 数据集根路径
    """
    dataset_root = Path(dataset_root)
    families = [f for f in dataset_root.iterdir() if f.is_dir() and f.name.startswith("family")]
    
    logger.info(f"找到 {len(families)} 个Family文件夹")
    
    for family in tqdm(families, desc="处理Family"):
        # 创建列表文件路径
        train_list = family / "train_wav_list.txt"
        test_list = family / "test_wav_list.txt"
        
        # 检查是否已经存在列表文件
        if train_list.exists() and test_list.exists():
            logger.debug(f"{family.name} 已存在WAV列表文件，跳过...")
            continue
        
        # 收集train目录下的所有WAV文件
        train_wavs = []
        train_dir = family / "train"
        if train_dir.exists():
            for speaker_dir in train_dir.iterdir():
                if speaker_dir.is_dir():
                    for wav_file in speaker_dir.glob("*.wav"):
                        # 使用相对于数据集根目录的路径
                        rel_path = wav_file.relative_to(dataset_root)
                        train_wavs.append(f"./{rel_path}")
        
        # 收集test目录下的所有WAV文件
        test_wavs = []
        test_dir = family / "test"
        if test_dir.exists():
            for speaker_dir in test_dir.iterdir():
                if speaker_dir.is_dir():
                    for wav_file in speaker_dir.glob("*.wav"):
                        # 使用相对于数据集根目录的路径
                        rel_path = wav_file.relative_to(dataset_root)
                        test_wavs.append(f"./{rel_path}")
        
        # 写入train列表文件
        if train_wavs:
            with open(train_list, "w") as f:
                f.write("\n".join(train_wavs))
            logger.info(f"为 {family.name} 创建了train列表，包含 {len(train_wavs)} 个WAV文件")
        else:
            logger.warning(f"{family.name} 的train目录中没有找到WAV文件")
        
        # 写入test列表文件
        if test_wavs:
            with open(test_list, "w") as f:
                f.write("\n".join(test_wavs))
            logger.info(f"为 {family.name} 创建了test列表，包含 {len(test_wavs)} 个WAV文件")
        else:
            logger.warning(f"{family.name} 的test目录中没有找到WAV文件")
    
    logger.info("所有Family的WAV列表文件生成完成!")

def main():
    parser = argparse.ArgumentParser(description='为每个Family生成WAV列表文件')
    parser.add_argument('--dataset_root', required=True, help='数据集根路径')
    
    args = parser.parse_args()
    
    # 执行处理
    generate_wav_lists(
        dataset_root=args.dataset_root
    )

if __name__ == "__main__":
    main()