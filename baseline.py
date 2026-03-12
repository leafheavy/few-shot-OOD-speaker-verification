import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

from dataloader import SpeakerEmbeddingDataset, create_dataloaders_for_families
from torch.utils.data import DataLoader
from pathlib import Path

from calculate_eer import calculate_eer

logger = logging.getLogger(__name__)

class InnerProductCalculator:
    def __init__(self, metric='inner_product'):
        """
        纯内积计算器
        
        Args:
            metric: 相似度度量方法，'inner_product'（内积）或 'cosine'（余弦相似度）
        """
        self.metric = metric
        self.W = None  # 训练集嵌入矩阵 [num_support, embedding_dim]
        self.support_labels = None
        self.train_speaker_ids = None
        
    def build_W_matrix(self, support_loader):
        """
        从train_loader构建嵌入矩阵W
        
        Args:
            support_loader: support set 数据加载器
            
        Returns:
            W: 嵌入矩阵 [num_support, embedding_dim]
        """
        all_embeddings = []
        all_labels = []
        all_speaker_ids = []
        
        with torch.no_grad():
            for batch in support_loader:
                embeddings, labels, speaker_ids, _ = batch
                all_embeddings.append(embeddings)
                all_labels.append(labels)
                all_speaker_ids.extend(speaker_ids)
        
        # 拼接所有训练嵌入
        self.W = torch.cat(all_embeddings, dim=0)  # [num_support, embedding_dim]
        self.support_labels = torch.cat(all_labels, dim=0)
        self.train_speaker_ids = all_speaker_ids
        
        # 如果需要余弦相似度，对嵌入进行归一化
        if self.metric == 'cosine':
            self.W = F.normalize(self.W, p=2, dim=1)
        
        # logger.info(f"构建嵌入矩阵 W: {self.W.shape}, samples of the support set: {len(self.support_labels)}")
        return self.W
    
    def compute_inner_products(self, test_loader, return_similarity_matrix=False):
        """
        计算测试嵌入与训练嵌入的内积
        
        Args:
            test_loader: 测试数据加载器
            return_similarity_matrix: 是否返回完整的相似度矩阵
            
        Returns:
            内积计算结果
        """
        if self.W is None:
            raise ValueError("必须先调用 build_W_matrix() 方法构建W矩阵")
        
        all_test_embeddings = []
        all_true_labels = []
        all_test_speaker_ids = []
        
        with torch.no_grad():
            for batch in test_loader:
                embeddings, labels, speaker_ids, _ = batch
                all_test_embeddings.append(embeddings) # 已由 pretrained model 提取得到的 embedding
                all_true_labels.append(labels) # 0~4 五个 speaker id 的映射
                all_test_speaker_ids.extend(speaker_ids) # 五个 speaker id
        
        # 拼接所有测试嵌入
        X_test = torch.cat(all_test_embeddings, dim=0)  # [num_test, embedding_dim]
        true_labels = torch.cat(all_true_labels, dim=0)
        
        # 如果需要余弦相似度，对测试嵌入进行归一化
        if self.metric == 'cosine':
            X_test = F.normalize(X_test, p=2, dim=1)
        
        # 计算内积相似度矩阵 [num_test, num_support]
        if self.metric == 'inner_product':
            similarity_matrix = torch.matmul(X_test, self.W.T)  # 内积[1,6](@ref)
        else:  # cosine
            similarity_matrix = torch.matmul(X_test, self.W.T)  # 已经归一化，内积=余弦相似度
        
        # logger.info(f"相似度矩阵形状: {similarity_matrix.shape}")
        
        # 将样本索引映射为说话人标签
        max_indices = torch.max(similarity_matrix, dim=1)[1]  # 样本索引 0~14
        predicted_labels = self.support_labels[max_indices]   # 说话人标签 0~4

        return {
            'similarity_matrix': similarity_matrix,
            'test_embeddings': X_test,
            'true_labels': true_labels,
            'test_speaker_ids': all_test_speaker_ids,
            'W_matrix': self.W,
            'support_labels': self.support_labels,
            'train_speaker_ids': self.train_speaker_ids,
            'max_similarities': torch.max(similarity_matrix, dim=1)[0],
            'prediction': predicted_labels,  # 现在范围是 0~4
            'true_labels': true_labels,       # 范围也是 0~4
            'test_speaker_ids': all_test_speaker_ids
        }

if __name__ == "__main__":
    with open("baseline_result_cosine.txt", "w") as f:
        # 初始化内积计算器
        # calculator = InnerProductCalculator(metric='inner_product')
        calculator = InnerProductCalculator(metric='cosine')
        
        # 存储小样本的 embedding 的文件夹
        dataset_root = Path('/Dataset/Voxceleb1/voxceleb1_5shot3way')
        
        support_loaders, test_loaders = create_dataloaders_for_families(dataset_root, batch_size=8, preload=True)

        families = [f for f in dataset_root.iterdir() if f.is_dir() and f.name.startswith("family")]

        total_eers = []
        total_err_num, total_test_num = 0, 0
        for family in tqdm(families):
            support_loader = support_loaders[family.name]
            test_loader = test_loaders[family.name]
        
        # # 选择第一个family进行示例
        # family_name = list(support_loaders.keys())[0]
        # support_loader = support_loaders[family_name]
        # test_loader = test_loaders[family_name]
        
            # 1. 构建W矩阵
            W_matrix = calculator.build_W_matrix(support_loader) # shape: (num_support, embedding_dim)
            
            # 2. 计算内积
            results = calculator.compute_inner_products(test_loader)
            
            pred = results['prediction'] # shape: (num_test,)
            true_labels = results['true_labels'] # shape: (num_test,)

            # 3. 统计错判的个数
            err_num = torch.count_nonzero(pred - true_labels)
            test_num = len(true_labels)

            total_err_num += err_num
            total_test_num += test_num

            err_ration = err_num / test_num * 100

            # 4. 计算EER
            # 获取相似度矩阵和真实标签
            similarity_matrix = results['similarity_matrix']  # [num_test, num_support]
            
            # 为EER计算准备数据：将多分类问题转换为二分类问题
            eer_scores = []
            eer_labels = []
            
            for i in range(len(true_labels)):
                true_speaker = true_labels[i].item()
                
                # 对于每个测试样本，找到其与真实说话人支持样本的最大相似度作为正例得分
                speaker_support_indices = (results['support_labels'] == true_speaker).nonzero(as_tuple=True)[0]
                if len(speaker_support_indices) > 0:
                    positive_score = torch.max(similarity_matrix[i, speaker_support_indices]).item()
                    eer_scores.append(positive_score)
                    eer_labels.append(1)  # 正例
                    
                # 找到与其他说话人的最大相似度作为负例得分
                imposter_support_indices = (results['support_labels'] != true_speaker).nonzero(as_tuple=True)[0]
                if len(imposter_support_indices) > 0:
                    negative_score = torch.max(similarity_matrix[i, imposter_support_indices]).item()
                    eer_scores.append(negative_score)
                    eer_labels.append(0)  # 负例
            
            # 计算EER
            if len(eer_scores) > 0:
                eer, eer_threshold = calculate_eer(np.array(eer_scores), np.array(eer_labels))
                total_eers.append(eer)
            else:
                eer = 0.0
                eer_threshold = 0.0
                total_eers.append(eer)

            f.write(f"针对 {family.name} 的错误率为: {err_ration:.2f} %, EER为: {eer*100:.2f} % (阈值: {eer_threshold:.4f})\n")
            f.flush()

        total_err_ration = total_err_num / total_test_num * 100
        mean_eer = np.mean(total_eers)
        f.write(f"\n=============================================\n")
        f.write(f"所有 families 的错误率为: {total_err_ration:.2f} %, 平均EER为: {mean_eer*100:.2f} %\n")
    print("所有结果已保存到 baseline_result_cosine.txt")
