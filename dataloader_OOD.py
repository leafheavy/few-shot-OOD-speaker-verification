import logging
import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VOX_SPK_PATTERN = re.compile(r"(id\d+)", re.IGNORECASE)

def infer_speaker_id(file_path: Path) -> str:
    """从 embedding 文件名中推断说话人 ID，优先匹配 VoxCeleb 的 idXXXXX 格式。"""
    stem = file_path.stem
    m = VOX_SPK_PATTERN.search(stem)
    if m:
        return m.group(1)
    if '_' in stem:
        return stem.rsplit('_', 1)[0]
    return stem

class SpeakerEmbeddingDataset(Dataset):
    def __init__(self, data_dir, split='train', preload=False, ood_flag=False):
        """
        Args:
            data_dir: family 目录路径data_dir: family 目录路径
            split: 'train' 或 'test'
            preload: 是否预加载嵌入到内存中
            ood_flag: 是否为OOD数据集 (OOD 样本标签固定为-1)
        """
        self.data_dir = Path(data_dir)
        self.split_dir = self.data_dir / 'embedding' / split
        self.preload = preload
        self.ood_flag = ood_flag
        
        # 检查目录是否存在
        if not self.split_dir.exists():
            raise ValueError(f"Directory {self.split_dir} does not exist")
        
        self.file_paths = sorted(self.split_dir.rglob('*.npy'))

        if not self.file_paths:
            logger.warning(f"No embedding files found in {self.split_dir}")
            self.empty = True
            return
        
        self.empty = False
        
        # 创建speaker到label的映射
        self.speaker_ids = [infer_speaker_id(path) for path in self.file_paths]
        unique_speakers = sorted(set(self.speaker_ids))
        self.speaker_to_label = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
        
        # 创建标签列表
        self.labels = [self.speaker_to_label[speaker_id] for speaker_id in self.speaker_ids]
        
        # 预加载所有嵌入到内存中
        self.default_embedding_dim = self._infer_embedding_dim()
        if self.preload:
            self.embeddings = [self._load_embedding(path) for path in self.file_paths]

    def _infer_embedding_dim(self) -> int:
        for file_path in self.file_paths:
            try:
                emb = np.load(file_path)
                if emb.ndim >= 1:
                    return int(emb.shape[-1])
            except Exception:
                continue
        return 192

    def _load_embedding(self, file_path: Path) -> torch.Tensor:
        try:
            embedding = np.load(file_path)
            return torch.from_numpy(embedding).float()
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return torch.zeros(self.default_embedding_dim)
    
    def __len__(self):
        if self.empty:
            return 0
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        if self.empty:
            raise IndexError("Dataset is empty")
            
        if self.preload:
            # 从内存中获取预加载的嵌入
            embedding = self.embeddings[idx]
        else:
            # 直接从磁盘加载嵌入
            embedding = self._load_embedding(self.file_paths[idx])

        label = -1 if self.ood_flag else self.labels[idx]
        speaker_id = self.speaker_ids[idx]
        
        return embedding, label, speaker_id, str(self.file_paths[idx])

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
    families = sorted(
        [f for f in dataset_root.iterdir() if f.is_dir() and f.name.startswith("family")],
        key=lambda p: p.name,
    )
    
    support_loaders = {}
    combined_test_loaders = {}

    if len(families) < 2:
        logger.warning("At least 2 families are required for OOD evaluation; got %d", len(families))
        return support_loaders, combined_test_loaders
    
    for idx, family in enumerate(families):
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
                pin_memory=True,
            )
            support_loaders[family_name] = support_dataloader
        except Exception as e:
            logger.error(f"Error creating train dataloader for {family_name}: {e}")
            continue
        
        # 创建测试集dataloader
        try:
            test_dataset = SpeakerEmbeddingDataset(family, split='test', preload=preload)
            ood_family = families[(idx + 1) % len(families)]
            ood_test_dataset = SpeakerEmbeddingDataset(
                ood_family, split='test', preload=preload, ood_flag=True
            )
                
            if test_dataset.empty or ood_test_dataset.empty:
                logger.warning(f"Skipping {family_name} test dataset (empty)")
                support_loaders.pop(family_name, None)
                continue

            combined_dataset = ConcatDataset([test_dataset, ood_test_dataset])

            combined_test_dataloader = DataLoader(
                combined_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            combined_test_loaders[family_name] = combined_test_dataloader
            logger.info(
                "OOD pairing: ID=%s (%d samples) -> OOD=%s (%d samples)",
                family_name,
                len(test_dataset),
                ood_family.name,
                len(ood_test_dataset),
            )

        except Exception as e:
            logger.error(f"Error creating test dataloader for {family_name}: {e}")
            # 如果测试集创建失败，我们也需要移除训练集的dataloader
            support_loaders.pop(family_name, None)
            continue
    
    return support_loaders, combined_test_loaders

def create_combined_dataloader(families, split='train', batch_size=8, preload=False):
    """
    创建包含所有families数据的dataloader
    
    Args:
        families: family路径列表
        split: 'train' 或 'test'
        batch_size: 批处理大小
        preload: 是否预加载嵌入到内存中
    
    Returns:
        combined_dataloader: 合并的DataLoader
    """
    datasets = []
    for family_path in families:
        try:
            dataset = SpeakerEmbeddingDataset(family_path, split=split, preload=preload)
            if not dataset.empty:
                datasets.append(dataset)
            else:
                logger.warning(f"Skipping {family_path.name} {split} dataset (empty)")
        except Exception as e:
            logger.error(f"Error creating dataset for {family_path}: {e}")
    
    # 检查是否有有效的数据集
    if not datasets:
        logger.error(f"No valid datasets found for {split}")
        return None
    
    # 合并所有数据集
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    
    # 创建dataloader
    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True,
    )
    
    logger.info(f"Combined {split} dataset: {len(combined_dataset)} samples, {len(dataloader)} batches")
    
    return dataloader

# 使用示例
if __name__ == "__main__":
    # 存储小样本的 embedding 的文件夹
    dataset_root = Path('/Dataset/Voxceleb1/voxceleb1_5shot3way')
    
    # 方法1: 为每个family创建独立的DataLoader
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
    
    # # 方法2: 创建合并所有families的DataLoader
    families = sorted(
        [f for f in dataset_root.iterdir() if f.is_dir() and f.name.startswith("family")],
        key=lambda p: p.name,
    )
    
    # combined_train_dataloader = create_combined_dataloader(families, split='train', batch_size=8, preload=True)
    # combined_test_dataloader = create_combined_dataloader(families, split='test', batch_size=8, preload=True)
    
    # # 使用合并的DataLoader
    # if combined_train_dataloader:
    #     for batch_idx, (embeddings, labels, speaker_id, file_paths) in enumerate(combined_train_dataloader):
    #         print(f"Combined Batch {batch_idx}: embeddings shape = {embeddings.shape}, labels = {labels}, speaker_id = {speaker_id}, file_paths = {file_paths}")
    #         # 在这里进行训练
    #         if batch_idx == 2:  # 只打印前几个batch
    #             break