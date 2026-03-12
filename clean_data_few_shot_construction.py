import os
import csv
import shutil
from collections import defaultdict

# 配置路径
meta_dir = "/data/voxceleb/meta"
audio_dir = "/data/voxceleb/audio"
output_base = "/data/voxceleb/5shot3way_clean_data"

# 配置文件
csv_file = "clean_data.csv"  # 修正变量名

# 数据结构
speaker_data = defaultdict(list)   # {speaker_id: [wav_paths]}
family_counter = 1                 # 当前家族编号

speakers_per_family = 5            # 每个家族的说话人数
shot_num = 3                       # 每个说话人训练集文件数

# 需要处理的列（避免处理包含特殊字符的无关列）
required_columns = (
    [f'user_id_{i}' for i in range(1, 6)] +
    [f'voice_id_{i}' for i in range(1, 6)] +
    ['user_id_test', 'voice_id_test']
)

# 第一阶段：从CSV文件收集所有speaker的wav文件
csv_path = os.path.join(meta_dir, csv_file)
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at {csv_path}")

# 以utf-8读取，自动替换无法解码的字符
with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.DictReader(f)
    
    # 只处理我们需要的列，跳过其他列
    for row in reader:
        # 处理5个参考语音
        for i in range(1, 6):
            user_id_col = f'user_id_{i}'
            voice_id_col = f'voice_id_{i}'
            
            # 仅当列存在且有值时处理
            if user_id_col in row and voice_id_col in row:
                user_id = row[user_id_col].strip()
                voice_file = row[voice_id_col].strip()
                
                if user_id and voice_file:
                    wav_path = os.path.join(audio_dir, voice_file)
                    if os.path.exists(wav_path):
                        speaker_data[user_id].append(wav_path)
        
        # 处理测试语音
        if 'user_id_test' in row and 'voice_id_test' in row:
            user_id_test = row['user_id_test'].strip()
            voice_id_test = row['voice_id_test'].strip()
            
            if user_id_test and voice_id_test:
                wav_path = os.path.join(audio_dir, voice_id_test)
                if os.path.exists(wav_path):
                    speaker_data[user_id_test].append(wav_path)

# 第二阶段：过滤掉文件数不足的speaker
valid_speakers = {}
for speaker_id, paths in speaker_data.items():
    # 去重
    unique_paths = list(set(paths))
    
    # 需要至少4个文件: 3个训练 + 1个测试
    if len(unique_paths) >= shot_num + 1:
        valid_speakers[speaker_id] = unique_paths

print(f"找到 {len(valid_speakers)} 个有效说话人 (每个至少有 {shot_num+1} 个音频文件)")

# 第三阶段：按家族分配并复制文件
current_family_speakers = []  # 当前家族中的speaker列表

def process_speaker(speaker_id, paths, family_dir):
    """处理单个speaker的文件分发"""
    # 创建目录
    train_dir = os.path.join(family_dir, "train", speaker_id)
    test_dir = os.path.join(family_dir, "test", speaker_id)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 按文件名排序确保一致性
    sorted_paths = sorted(paths)
    
    # 分发文件（前shot_num个训练，其余测试）
    for i, src_path in enumerate(sorted_paths):
        filename = os.path.basename(src_path)
        if i < shot_num:
            dest_path = os.path.join(train_dir, filename)
        else:
            dest_path = os.path.join(test_dir, filename)
        
        # 避免重复复制
        if not os.path.exists(dest_path):
            shutil.copy2(src_path, dest_path)

# 按speaker_id排序确保一致性
all_speakers = sorted(valid_speakers.keys())

for speaker_id in all_speakers:
    if len(current_family_speakers) >= speakers_per_family:
        # 创建家族目录
        family_dir = os.path.join(output_base, f"family{family_counter:05d}")
        os.makedirs(family_dir, exist_ok=True)
        
        # 处理当前家族的所有speaker
        for sid in current_family_speakers:
            process_speaker(sid, valid_speakers[sid], family_dir)
        
        # 重置并计数
        current_family_speakers = []
        family_counter += 1
    
    current_family_speakers.append(speaker_id)

# 处理最后一个不完整的家族
if current_family_speakers:
    family_dir = os.path.join(output_base, f"family{family_counter:05d}")
    os.makedirs(family_dir, exist_ok=True)
    for sid in current_family_speakers:
        process_speaker(sid, valid_speakers[sid], family_dir)

print(f"数据集构建完成！共处理了 {len(valid_speakers)} 个有效speaker, 组成 {family_counter} 个家族。")