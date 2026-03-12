import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
import numpy as np
import logging
import os
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dataloader_OOD import (SpeakerEmbeddingDataset, create_combined_dataloader, create_dataloaders_for_families, infer_speaker_id)

class SpeakerEmbeddingDataset(Dataset):
    def __init__(self, data_dir, split='train', preload=False, ood_flag=False):
        """
        Args:
            data_dir: 数据集根目录 (如 /data/xz-data/5shot3way/family00001)
            split: 'train' 或 'test'
            preload: 是否预加载嵌入到内存中
            ood_flag: 是否为OOD数据集
        """
        self.data_dir = Path(data_dir)
        self.split_dir = self.data_dir / 'embedding' / split
        self.preload = preload
        self.ood_flag = ood_flag

        # 检查目录是否存在
        if not self.split_dir.exists():
            raise ValueError(f"Directory {self.split_dir} does not exist")
        
        # 收集所有embedding文件路径
        self.file_paths = list(self.split_dir.glob('*.npy'))
        
        # 检查是否有任何嵌入文件
        if not self.file_paths:
            logger.warning(f"No embedding files found in {self.split_dir}")
            self.empty = True
            return
        
        self.empty = False
        
        # 从对应的wav文件路径中提取speaker_id
        self.speaker_ids = []
        self.valid_file_paths = []
        
        for file_path in self.file_paths:
            # 获取npy文件的GUID名称（不含扩展名）
            guid = file_path.stem
            
            # 查找对应的wav文件路径来确定speaker_id
            wav_search_pattern = f"**/{guid}.wav"
            wav_files = list(self.data_dir.glob(wav_search_pattern))
            
            if wav_files:
                # 从wav文件路径中提取speaker_id
                wav_path = wav_files[0]
                # wav文件路径格式: .../familyXXXXX/{split}/{speaker_id}/{guid}.wav
                # 所以speaker_id是wav文件父目录的名称
                speaker_id = wav_path.parent.name
                self.speaker_ids.append(speaker_id)
                self.valid_file_paths.append(file_path)
            else:
                logger.warning(f"Could not find corresponding wav file for {file_path}")
        
        # 如果没有找到有效的文件
        if not self.valid_file_paths:
            logger.warning(f"No valid embedding files with corresponding wav files found in {self.split_dir}")
            self.empty = True
            return
        
        # 创建speaker到label的映射
        unique_speakers = sorted(set(self.speaker_ids))
        self.speaker_to_label = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
        
        # 创建标签列表
        self.labels = [self.speaker_to_label[speaker_id] for speaker_id in self.speaker_ids]
        
        # 预加载所有嵌入到内存中
        if self.preload:
            self.embeddings = []
            for file_path in self.valid_file_paths:
                try:
                    embedding = np.load(file_path)
                    self.embeddings.append(torch.from_numpy(embedding).float())
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    # 添加一个空的tensor作为占位符
                    self.embeddings.append(torch.zeros(192))  # 假设嵌入维度为192
            # logger.info(f"Preloaded {len(self.embeddings)} embeddings into memory")
    
    def __len__(self):
        if self.empty:
            return 0
        return len(self.valid_file_paths)
    
    def __getitem__(self, idx):
        if self.empty:
            raise IndexError("Dataset is empty")
            
        if self.preload:
            # 从内存中获取预加载的嵌入
            embedding = self.embeddings[idx]
        else:
            # 从磁盘读取嵌入
            file_path = self.valid_file_paths[idx]
            try:
                embedding = np.load(file_path)
                embedding = torch.from_numpy(embedding).float()
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                # 返回一个空的tensor作为占位符
                embedding = torch.zeros(192)  # 假设嵌入维度为192
        
        # 对于OOD数据集，标签设为-1
        if self.ood_flag:
            label = -1
        else:
            label = self.labels[idx]

        speaker_id = self.speaker_ids[idx]
        
        return embedding, label, speaker_id, str(self.valid_file_paths[idx])

def create_dataloaders_for_families(dataset_root, batch_size=8, preload=False):
    """
    为所有families创建DataLoader
    
    Args:
        dataset_root: 数据集根目录
        batch_size: 批处理大小
        preload: 是否预加载嵌入到内存中
    
    Returns:
        support_loaders: 每个family的训练DataLoader字典
        test_loaders: 每个family的测试DataLoader字典
    """
    dataset_root = Path(dataset_root)
    families = [f for f in dataset_root.iterdir() if f.is_dir() and f.name.startswith("family")]
    assert len(families) > 0, f"No families found in {dataset_root}"
    
    support_loaders = {}
    combined_test_loaders = {}
    
    for idx, family in tqdm(enumerate(families), desc="Constructing Few Shot Dataset"):
        family_name = family.name
        
        # 创建训练集dataloader
        try:
            train_dataset = SpeakerEmbeddingDataset(family, split='train', preload=preload)
            if train_dataset.empty:
                logger.warning(f"Skipping {family_name} train dataset (empty)")
                continue
                
            support_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            support_loaders[family_name] = support_dataloader
        except Exception as e:
            logger.error(f"Error creating train dataloader for {family_name}: {e}")
            continue
        
        # 创建测试集dataloader
        try:
            test_dataset = SpeakerEmbeddingDataset(family, split='test', preload=preload)
            if idx + 1 < len(families):
                # 检查下一个family的test数据是否为空，以防止跨family错误
                ood_test_dataset = SpeakerEmbeddingDataset(families[idx+1], split='test', preload=preload, ood_flag=True)
            else:
                ood_test_dataset = SpeakerEmbeddingDataset(families[0], split='test', preload=preload, ood_flag=True)
                
            if test_dataset.empty or ood_test_dataset.empty:
                logger.warning(f"Skipping {family_name} test dataset (empty)")
                # 如果测试集为空，我们也需要移除训练集的dataloader
                if family_name in support_loaders:
                    del support_loaders[family_name]
                continue

            combined_dataset = ConcatDataset([test_dataset, ood_test_dataset])
                
            combined_test_dataloader = DataLoader(
                combined_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            combined_test_loaders[family_name] = combined_test_dataloader
        except Exception as e:
            logger.error(f"Error creating test dataloader for {family_name}: {e}")
            # 如果测试集创建失败，我们也需要移除训练集的dataloader
            if family_name in support_loaders:
                del support_loaders[family_name]
            continue
        
        # logger.info(f"Family: {family_name}")
        # logger.info(f"Train samples: {len(train_dataset)}, batches: {len(support_dataloader)}")
        # logger.info(f"Test samples: {len(test_dataset)}, batches: {len(test_dataloader)}")
    
    return support_loaders, combined_test_loaders

# 使用示例
if __name__ == "__main__":
    # 存储小样本的 embedding 的文件夹
    dataset_root = Path('/data/xz-data/5shot3way')
    
    # 为每个family创建独立的DataLoader
    support_loaders, test_loaders = create_dataloaders_for_families(dataset_root, batch_size=8, preload=True)
    
    # 打印处理结果
    logger.info(f"Successfully processed {len(support_loaders)} families")
    
    # 使用示例: 访问特定family的DataLoader
    if support_loaders:
        family_name = list(support_loaders.keys())[0]  # 获取第一个family
        support_loader = support_loaders[family_name]
        test_loader = test_loaders[family_name]
        
        # 遍历数据
        for batch_idx, (embeddings, labels, speaker_id, file_paths) in enumerate(support_loader):
            print(f"Batch {batch_idx}: embeddings shape = {embeddings.shape}, labels = {labels}, speaker_id = {speaker_id}, file_paths = {file_paths}")
            # 在这里进行训练
            if batch_idx == 2:  # 只打印前几个batch
                break
