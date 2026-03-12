import os
import shutil
import random
from tqdm import tqdm

def construct_few_shot_dataset(
    source_root: str,
    output_root: str,
    family_size: int = 5,
    train_samples_per_member: int = 3,
    random_seed: int = 42
):
    """
    构建家庭语音小样本学习数据集
    
    Args:
        source_root: 原始数据集根路径 (E:/benchmark/audio/voxceleb1)
        output_root: 输出数据集根路径 (E:/benchmark/audio/voxceleb1_few_shot)
        family_size: 每个家庭成员数量 (默认5)
        train_samples_per_member: 每个成员训练样本数 (默认3)
        random_seed: 随机种子保证可复现
    """
    # 设置随机种子
    random.seed(random_seed)
    print(f"Using random seed: {random_seed}")
    
    # 创建输出目录
    os.makedirs(output_root, exist_ok=True)
    
    # 获取所有说话人ID
    speaker_ids = [
        d for d in os.listdir(source_root) 
        if os.path.isdir(os.path.join(source_root, d)) and d.startswith("id")
    ]
    speaker_ids.sort()
    print(f"Found {len(speaker_ids)} speakers")
    
    # 按家庭分组
    families = []
    for i in range(0, len(speaker_ids), family_size):
        if i + family_size <= len(speaker_ids):
            families.append(speaker_ids[i:i+family_size])
        else:
            print(f"Skipping incomplete family with {len(speaker_ids) % family_size} members")
    
    print(f"Created {len(families)} complete families")
    
    # 处理每个家庭
    for family_idx, family in enumerate(tqdm(families, desc="Processing families")):
        # 创建家庭目录 (family0001, family0002, ...)
        family_name = f"family{family_idx+1:04d}"
        family_path = os.path.join(output_root, family_name)
        os.makedirs(family_path, exist_ok=True)
        
        # 创建家庭内的train/test目录
        train_family_path = os.path.join(family_path, "train")
        test_family_path = os.path.join(family_path, "test")
        os.makedirs(train_family_path, exist_ok=True)
        os.makedirs(test_family_path, exist_ok=True)
        
        # 处理家庭中的每个成员
        for speaker_id in family:
            # 创建成员目录（不包含子文件夹）
            train_speaker_path = os.path.join(train_family_path, speaker_id)
            test_speaker_path = os.path.join(test_family_path, speaker_id)
            os.makedirs(train_speaker_path, exist_ok=True)
            os.makedirs(test_speaker_path, exist_ok=True)
            
            # 收集该说话人所有wav文件（忽略子文件夹结构）
            all_wav_files = []
            speaker_dir = os.path.join(source_root, speaker_id)
            
            # 遍历所有子文件夹收集wav
            for subfolder in os.listdir(speaker_dir):
                subfolder_path = os.path.join(speaker_dir, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue
                
                # 收集子文件夹中的wav文件
                for wav_file in os.listdir(subfolder_path):
                    if wav_file.lower().endswith(".wav"):
                        all_wav_files.append(os.path.join(subfolder_path, wav_file))
            
            # 检查样本数量
            if len(all_wav_files) < train_samples_per_member:
                raise ValueError(
                    f"Speaker {speaker_id} has only {len(all_wav_files)} samples, "
                    f"need at least {train_samples_per_member}"
                )
            
            # 随机选择训练样本
            selected = random.sample(all_wav_files, train_samples_per_member)
            remaining = [f for f in all_wav_files if f not in selected]
            
            # 复制训练样本（扁平化处理）
            for wav_path in selected:
                # 生成唯一文件名：speaker_id_序号.wav
                filename = f"{speaker_id}_{len(os.listdir(train_speaker_path)) + 1:03d}.wav"
                shutil.copy2(
                    wav_path,
                    os.path.join(train_speaker_path, filename)
                )
            
            # 复制测试样本（扁平化处理）
            for wav_path in remaining:
                # 生成唯一文件名：speaker_id_序号.wav
                filename = f"{speaker_id}_{len(os.listdir(test_speaker_path)) + 1:03d}.wav"
                shutil.copy2(
                    wav_path,
                    os.path.join(test_speaker_path, filename)
                )
    
    # 生成数据集统计报告
    report = f"""
    ====== Dataset Construction Report ======
    Source: {source_root}
    Output: {output_root}
    Total families: {len(families)}
    Family size: {family_size}
    Train samples per member: {train_samples_per_member}
    
    Directory Structure (FLATTENED):
    {output_root}/
    ├── family0001/
    │   ├── train/
    │   │   ├── id10001/  (3 wav files - flat structure)
    │   │   ├── id10002/  (3 wav files)
    │   │   └── ... 
    │   └── test/
    │       ├── id10001/  (remaining wav - flat structure)
    │       └── ... 
    ├── family0002/
    │   ├── train/
    │   └── test/
    └── ...
    
    Total speakers: {len(families) * family_size}
    ========================================
    """
    print(report)
    
    # 保存报告到文件
    with open(os.path.join(output_root, "dataset_report.txt"), "w") as f:
        f.write(report)

if __name__ == "__main__":
    # 配置路径 (根据实际路径)
    SOURCE_ROOT = r"E:/benchmark/audio/voxceleb1"
    OUTPUT_ROOT = r"E:/benchmark/audio/voxceleb1_few_shot"
    
    # 执行数据集构建 (5 shot 3 ways)
    construct_few_shot_dataset(
        source_root=SOURCE_ROOT,
        output_root=OUTPUT_ROOT,
        family_size=5,
        train_samples_per_member=3
    )