import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, roc_curve

from dataloader_OOD import SpeakerEmbeddingDataset, create_dataloaders_for_families
from torch.utils.data import DataLoader
from pathlib import Path

from calculate_eer import calculate_eer

logger = logging.getLogger(__name__)

def evaluate_closed_set_and_open_set(results):
    """
    评估闭集分类与 open-set 检测指标。

    Args:
        results: compute_inner_products 的输出字典
    Returns:
        metrics: 指标字典
    """
    true_labels = results['true_labels']
    pred_labels = results['prediction']
    confidence_scores = results['max_similarities']

    # ID 样本: 标签 >= 0; OOD 样本: 标签 < 0
    id_mask = true_labels >= 0
    ood_mask = true_labels < 0

    id_total = int(torch.sum(id_mask).item())
    ood_total = int(torch.sum(ood_mask).item())

    id_correct = 0
    id_false_reject = 0
    if id_total > 0:
        id_preds = pred_labels[id_mask]
        id_trues = true_labels[id_mask]
        id_correct = int(torch.sum(id_preds == id_trues).item())
        id_false_reject = int(torch.sum(id_preds < 0).item())

    ood_false_accept = 0
    if ood_total > 0:
        # OOD 被预测为某个 ID 标签（>=0）即 false accept
        ood_preds = pred_labels[ood_mask]
        ood_false_accept = int(torch.sum(ood_preds >= 0).item())

    id_acc = (id_correct / id_total) if id_total > 0 else 0.0
    id_false_reject_rate = (id_false_reject / id_total) if id_total > 0 else 0.0
    ood_false_accept_rate = (ood_false_accept / ood_total) if ood_total > 0 else 0.0

    # OOD 检测：1=ID，0=OOD
    ood_binary_labels = (true_labels >= 0).int().cpu().numpy()
    conf_np = confidence_scores.cpu().numpy()

    auroc = None
    fpr95 = None
    if len(np.unique(ood_binary_labels)) > 1:
        auroc = float(roc_auc_score(ood_binary_labels, conf_np))
        fpr, tpr, _ = roc_curve(ood_binary_labels, conf_np)
        idx_95 = np.where(tpr >= 0.95)[0]
        fpr95 = float(fpr[idx_95[0]]) if len(idx_95) > 0 else 1.0

    return {
        'id_total': id_total,
        'ood_total': ood_total,
        'id_acc': id_acc,
        'id_false_reject_rate': id_false_reject_rate,
        'ood_false_accept_rate': ood_false_accept_rate,
        'auroc': auroc,
        'fpr95': fpr95,
    }

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
    
    def compute_inner_products(self, test_loader, threshold=0.4, return_similarity_matrix=False):
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
        
        # 计算内积相似度矩阵 [num_test, num_support]
        if self.metric == 'inner_product':
            similarity_matrix = torch.matmul(X_test, self.W.T)  # 内积[1,6](@ref)
        else:  # cosine
            X_test = F.normalize(X_test, p=2, dim=1)
            similarity_matrix = torch.matmul(X_test, self.W.T)  # 已经归一化，内积=余弦相似度
        # 对每一行进行 Softmax 归一化
        similarity_matrix = F.softmax(similarity_matrix, dim=1)
        # logger.info(f"相似度矩阵形状: {similarity_matrix.shape}")
        
        # 将样本索引映射为说话人标签
        # max_indices = torch.max(similarity_matrix, dim=1)[1]  # 样本索引 0~14
        max_values, max_indices = torch.max(similarity_matrix, dim=1)
    
        predicted_labels = torch.full_like(true_labels, -1)
        valid_mask = max_values >= threshold
        if valid_mask.any():
            predicted_labels[valid_mask] = self.support_labels[max_indices[valid_mask]]

        # predicted_labels = self.support_labels[max_indices]   # 说话人标签 0~4

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
    with open("baseline_result_OOD_cosine.txt", "w") as f:
        # 初始化内积计算器
        # calculator = InnerProductCalculator(metric='inner_product')
        calculator = InnerProductCalculator(metric='cosine')
        
        # 存储小样本的 embedding 的文件夹
        dataset_root = Path('/Dataset/Voxceleb1/voxceleb1_5shot3way')
        
        support_loaders, test_loaders = create_dataloaders_for_families(dataset_root, batch_size=8, preload=True)

        families = [f for f in dataset_root.iterdir() if f.is_dir() and f.name.startswith("family")]

        total_eers = []
        total_id_correct = 0
        total_id_num = 0
        total_id_false_reject = 0
        total_ood_false_accept = 0
        total_ood_num = 0

        total_auroc = []
        total_fpr95 = []

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
            
            # 3. 闭集 + open-set 评估
            metrics = evaluate_closed_set_and_open_set(results)

            total_id_num += metrics['id_total']
            total_ood_num += metrics['ood_total']
            total_id_correct += int(metrics['id_acc'] * metrics['id_total'])
            total_id_false_reject += int(metrics['id_false_reject_rate'] * metrics['id_total'])
            total_ood_false_accept += int(metrics['ood_false_accept_rate'] * metrics['ood_total'])

            if metrics['auroc'] is not None:
                total_auroc.append(metrics['auroc'])
            if metrics['fpr95'] is not None:
                total_fpr95.append(metrics['fpr95'])

            # 4. 计算EER
            # 获取相似度矩阵和真实标签
            similarity_matrix = results['similarity_matrix']  # [num_test, num_support]
            true_labels = results['true_labels']

            # 为EER计算准备数据：将多分类问题转换为二分类问题
            eer_scores = []
            eer_labels = []
            
            for i in range(len(true_labels)):
                true_speaker = true_labels[i].item()
                
                # OOD 样本不参与该 proxy-EER
                if true_speaker < 0:
                    continue

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

            id_acc_pct = metrics['id_acc'] * 100
            id_fr_pct = metrics['id_false_reject_rate'] * 100
            ood_fa_pct = metrics['ood_false_accept_rate'] * 100

            if metrics['auroc'] is not None and metrics['fpr95'] is not None:
                f.write(
                    f"针对 {family.name} | ID准确率: {id_acc_pct:.2f}% | ID错分为OOD: {id_fr_pct:.2f}% | "
                    f"OOD错分为ID: {ood_fa_pct:.2f}% | OOD AUROC: {metrics['auroc']:.4f} | "
                    f"FPR@95: {metrics['fpr95']:.4f} | proxy-EER: {eer*100:.2f}% (阈值: {eer_threshold:.4f})\n"
                )
            else:
                f.write(
                    f"针对 {family.name} | ID准确率: {id_acc_pct:.2f}% | ID错分为OOD: {id_fr_pct:.2f}% | "
                    f"OOD错分为ID: {ood_fa_pct:.2f}% | OOD AUROC/FPR@95: 数据不足 | "
                    f"proxy-EER: {eer*100:.2f}% (阈值: {eer_threshold:.4f})\n"
                )
            f.flush()

        total_id_acc = (total_id_correct / total_id_num * 100) if total_id_num > 0 else 0.0
        total_id_false_reject_rate = (total_id_false_reject / total_id_num * 100) if total_id_num > 0 else 0.0
        total_ood_false_accept_rate = (total_ood_false_accept / total_ood_num * 100) if total_ood_num > 0 else 0.0
        mean_eer = np.mean(total_eers)
        mean_auroc = np.mean(total_auroc) if total_auroc else 0.0
        mean_fpr95 = np.mean(total_fpr95) if total_fpr95 else 0.0
        f.write(f"\n=============================================\n")
        f.write(f"ID总体准确率: {total_id_acc:.2f}%\n")
        f.write(f"ID错分为OOD错误率: {total_id_false_reject_rate:.2f}%\n")
        f.write(f"OOD错分为ID错误率: {total_ood_false_accept_rate:.2f}%\n")
        f.write(f"OOD平均AUROC: {mean_auroc:.4f}\n")
        f.write(f"OOD平均FPR@95: {mean_fpr95:.4f}\n")
        f.write(f"proxy-EER均值: {mean_eer*100:.2f}%\n")
    print("所有结果已保存到 txt 文件中。")
