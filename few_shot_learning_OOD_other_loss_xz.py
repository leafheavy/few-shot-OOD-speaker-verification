import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, roc_curve

from xz_dataloader_OOD import SpeakerEmbeddingDataset, create_dataloaders_for_families
from torch.utils.data import DataLoader
from pathlib import Path

from calculate_eer import calculate_eer

from loss_functions import AngularPenaltySMLoss

logger = logging.getLogger(__name__)

# 2.Entropy Regularization
@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

# 3.引入 Cosine 的分类头
class CosineClassifier(nn.Module):
    def __init__(self, weight, bias):
        super(CosineClassifier, self).__init__()
        self.weight = nn.Parameter(weight)  # 形状: (num_classes, embedding_dim)
        self.bias = nn.Parameter(bias)      # 形状: (num_classes,)
        
    def forward(self, x):
        # 对 query 进行 L2 归一化, w 已经归一化过了
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        
        # 计算余弦相似度 (归一化输入与已归一化权重的点积)
        cosine_similarity = torch.mm(x_norm, self.weight.t())
        
        # 加上偏置项
        output = cosine_similarity + self.bias
        
        return output

class FewShotLearning:
    def __init__(self, device):
        """
        小样本学习器
        """
        self.classifier = None
        self.W = None  # 支撑集嵌入矩阵 [num_support, embedding_dim]
        self.b = None # 偏置项
        self.support_labels = None
        self.support_speaker_ids = None
        self.num_classes = 0 # 支持集中的类别数
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device  # 保存设备信息
    
    # 1.Good Initialization for Few-Shot Learning
    # 针对每个 speaker, 计算并归一化均值向量的函数
    def initialization(self, support_loader):
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
                # 将数据移动到指定设备
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)

                # print(f"Batch embeddings shape: {embeddings.shape}")  # 添加此行：检查每个batch的嵌入形状
                all_embeddings.append(embeddings)
                all_labels.append(labels)
                all_speaker_ids.extend(speaker_ids)
        
        self.support_labels = torch.cat(all_labels, dim=0)
        unique_labels = torch.unique(self.support_labels)
        self.num_classes = len(unique_labels)
        
        # 拼接所有支撑集的嵌入
        self.W = torch.cat(all_embeddings, dim=0) # [num_support, embedding_dim]

        # 计算均值并归一化
        # 按标签分组并计算每个类的原型, i.e., 即将每个 speaker 的多条样本均值作为该 speaker 的表示
        # 计算均值并归一化
        prototypes = []
        for label in unique_labels:
            # 重点检查这一行：all_labels 应该是之前所有batch的labels拼接起来的一维张量
            class_mask = (self.support_labels == label)  # 确保这里用的是拼接后的所有标签
            class_embeddings = self.W[class_mask]  # 获取该类别所有样本的嵌入
            class_prototype = torch.mean(class_embeddings, dim=0) # 计算原型
            prototypes.append(class_prototype)
        
        prototypes_tensor = torch.stack(prototypes)  # 将原型列表堆叠成张量
        self.W = F.normalize(prototypes_tensor, p=2, dim=1).to(self.device)

        self.b = torch.zeros(self.num_classes).to(self.device)  # 动态初始化偏置项

        self.support_speaker_ids = all_speaker_ids
        
        # logger.info(f"构建嵌入矩阵 W: {self.W.shape}, samples of the support set: {len(self.support_labels)}")
        # return self.W
        return

    def learning(self, support_loader, Epochs=100, lr=0.001):
        if self.W is None:
            raise ValueError("必须先调用 initialization() 方法构建W矩阵")

        # criterion = nn.CrossEntropyLoss()
        in_features = self.W.shape[1]
        out_features = self.W.shape[0]

        self.criterion = AngularPenaltySMLoss(in_features, out_features, loss_type='cosface').to(self.device) # loss_type in ['arcface', 'sphereface', 'cosface']
        
        # 将初始化得到的原型权重赋给损失函数的分类器
        with torch.no_grad():
            self.criterion.fc.weight.data = self.W.clone()
        
        self.classifier = self.criterion.fc.to(self.device)
        self.s = self.criterion.s  # 保存尺度因子用于推理

        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)

        self.classifier.train()

        for epoch in range(Epochs):
            epoch_loss = 0.0
            for batch in support_loader:
                embeddings, labels, speaker_ids, _ = batch
                # 将数据移动到设备
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)

                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                loss, logits = self.criterion(embeddings, labels, return_logits=True)  # 交叉熵损失
                entropy_reg = softmax_entropy(logits).mean()  # 熵正则化
                total_loss = loss + 0.01 * entropy_reg  # 总损失
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            avg_loss = epoch_loss / len(support_loader)
            # if (epoch + 1) % 10 == 0 or epoch == 0:
                # logger.info(f"Epoch [{epoch+1}/{Epochs}], Loss: {avg_loss:.4f}")
        # logger.info(f"Epoch [{epoch+1}/{Epochs}], Loss: {avg_loss:.4f}")
        
        # 更新后的 W 和 b
        self.W = self.classifier.weight.data
        # self.b = self.classifier.bias.data

        return 

    def calculate_result(self, test_loader):
        """
        测试小样本学习的效果
        
        Args:
            test_loader: 测试数据加载器
        Returns:
            内积计算结果
        """
        if self.W is None:
            raise ValueError("必须先调用 initialization() 方法构建W矩阵")
        
        all_test_embeddings = []
        all_true_labels = []
        all_test_speaker_ids = []
        all_pred_labels = []
        all_logits = []

        self.classifier.eval()  # 设置为评估模式
        
        with torch.no_grad():
            for batch in test_loader:
                embeddings, labels, speaker_ids, _ = batch
                # 将数据移动到设备
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)

                # 1. 归一化输入
                embeddings_norm = F.normalize(embeddings, p=2, dim=1)
                
                # 2. 归一化权重（分类器权重）
                w_norm = F.normalize(self.classifier.weight, p=2, dim=1)

                # 3. 计算余弦相似度并乘以尺度因子
                cosine_sim = torch.mm(embeddings_norm, w_norm.t())
                logits = self.s * cosine_sim  # 应用尺度因子

                predicted_labels = torch.argmax(logits, dim=1)

        #         all_test_embeddings.append(embeddings) # 已由 pretrained model 提取得到的 embedding
        #         all_true_labels.append(labels) # 0~4 五个 speaker id 的映射
        #         all_test_speaker_ids.extend(speaker_ids) # 五个 speaker id
        #         all_pred_labels.append(predicted_labels) # 预测的标签 0~4
        #         all_logits.append(logits)  # 累积logits
        
        # # 计算相似度矩阵用于EER计算
        # X_test = torch.cat(all_test_embeddings, dim=0)  # [num_test, embedding_dim]
        # # 对测试嵌入进行归一化
        # X_test_normalized = F.normalize(X_test, p=2, dim=1)
        # similarity_matrix = torch.mm(X_test_normalized, self.W.t()) # [num_test, num_speaker]

        # true_labels = torch.cat(all_true_labels, dim=0)
        # all_logits_tensor = torch.cat(all_logits, dim=0)
        # predictions = torch.cat(all_pred_labels, dim=0)
                # 将数据移回CPU用于后续处理
                all_test_embeddings.append(embeddings.cpu())
                all_true_labels.append(labels.cpu())
                all_test_speaker_ids.extend(speaker_ids)
                all_pred_labels.append(predicted_labels.cpu())
                all_logits.append(logits.cpu())
        
        # 在CPU上处理结果
        X_test = torch.cat(all_test_embeddings, dim=0)
        X_test_normalized = F.normalize(X_test, p=2, dim=1)
        # 将W矩阵移到CPU用于计算相似度
        W_cpu = self.W.cpu()
        similarity_matrix = torch.mm(X_test_normalized, W_cpu.t())

        true_labels = torch.cat(all_true_labels, dim=0)
        all_logits_tensor = torch.cat(all_logits, dim=0)
        predictions = torch.cat(all_pred_labels, dim=0)

        return {
            'similarity_matrix': similarity_matrix,
            'test_embeddings': X_test,
            'logits': all_logits_tensor,  # 返回 logits
            'W_matrix': self.W,
            'support_labels': self.support_labels,
            'support_speaker_ids': self.support_speaker_ids,
            'confidence_scores': torch.max(similarity_matrix, dim=1)[0],
            'prediction': predictions,  # 现在范围是 0~4
            'true_labels': true_labels,       # 范围也是 0~4
            'test_speaker_ids': all_test_speaker_ids
        }

if __name__ == "__main__":
    device = 'cuda:1' if torch.cuda.device_count() > 0 else 'cpu'

    for ood_threshold in [0.3, 0.4, 0.5, 0.6, 0.65]:
        with open(f"xiaozhi_result/5shot3way_3/few_shot_learning_OOD_cosface_{ood_threshold}.txt", "w") as f:
            # 初始化内积计算器
            mymodel = FewShotLearning(device=device)
            
            # 存储小样本的 embedding 的文件夹
            dataset_root = Path('/Dataset/xiaozhi/5shot3way_3')
            
            support_loaders, test_loaders = create_dataloaders_for_families(dataset_root, batch_size=128, preload=True)

            families = [f for f in dataset_root.iterdir() if f.is_dir() and f.name.startswith("family")]

            total_eers = []
            total_id_err_num = 0
            total_id_false_reject_num = 0
            total_id_class_err_num = 0
            total_id_test_num = 0
            total_ood_err_num = 0
            total_ood_test_num = 0

            for family in tqdm(families):
                support_loader = support_loaders[family.name]
                test_loader = test_loaders[family.name]
            
            # # 选择第一个family进行示例
            # family_name = list(support_loaders.keys())[0]
            # support_loader = support_loaders[family_name]
            # test_loader = test_loaders[family_name]
            
                # 1. 构建W矩阵
                mymodel.initialization(support_loader) # shape: (num_support, embedding_dim)
                
                # 2. 小样本学习
                mymodel.learning(support_loader, Epochs=100, lr=0.001)

                # 3. 测试
                results = mymodel.calculate_result(test_loader)
                
                pred = results['prediction'] # shape: (num_test,)
                true_labels = results['true_labels'] # shape: (num_test,)
                similarity_matrix = results['similarity_matrix']  # 获取 logits

                # 4. 计算 ER
                confidence_scores = torch.max(similarity_matrix, dim=1)[0].numpy()

                # ID 样本掩码
                id_mask = (true_labels >= 0)
                ood_mask = ~id_mask

                # 3. 计算 ID 分类错误率
                if torch.any(id_mask):
                    # 取出 ID 的真实标签（tensor）
                    id_true_labels = true_labels[id_mask]  # tensor, shape [n_id,]

                    # 取出对应行的 softmax 概率矩阵（或你返回的 similarity_matrix）
                    probs_id = similarity_matrix[id_mask, :]  # tensor [n_id, num_classes]

                    # 每个样本的最大 softmax 概率（MSP）
                    # torch.max 返回 (values, indices)；取 values
                    max_vals, max_idxs = torch.max(probs_id, dim=1)  # both tensors shape [n_id,]

                    # 构造两类 mask：被拒（max < T）与被接受（max >= T）
                    id_reject_mask = (max_vals < ood_threshold)   # tensor bool [n_id,]
                    id_accept_mask = ~id_reject_mask

                    # 1) ID False Rejects: 错分为 OOD
                    id_false_reject_num = int(torch.count_nonzero(id_reject_mask).item())

                    total_id_false_reject_num += id_false_reject_num

                    # 2) ID 内部分类错误
                    id_class_err_num = 0
                    id_accept_num = int(torch.count_nonzero(id_accept_mask).item())
                    if id_accept_num > 0:
                        id_pred_labels_all = results['prediction'][id_mask]  # tensor [n_id,]
                        # 只保留被接受的那些样本
                        id_pred_labels_accept = id_pred_labels_all[id_accept_mask]
                        id_true_accept = id_true_labels[id_accept_mask]

                        # 如果 predictions 是 prototype 索引，则需要 map 回原始标签：
                        # support_labels_tensor = self.support_labels  # 例如 tensor([label0, label1, ...])
                        # id_pred_labels_accept = support_labels_tensor[id_pred_labels_accept]

                        # 计算 ID 内部分类错误数
                        id_class_err_num = int(torch.count_nonzero(id_pred_labels_accept != id_true_accept).item())

                        # 累加分类错误统计（只计入那些被接受的 ID）
                        total_id_class_err_num += id_class_err_num
                        
                    else:
                        # 若没有被接受的样本，则分类错误数为 0，但仍应记录 id_accept_num = 0
                        id_class_err_num = 0

                    id_test_num = len(id_true_labels)
                    total_id_test_num += id_test_num

                    id_false_reject_rate = id_false_reject_num / id_test_num * 100
                    id_class_err_rate = id_class_err_num / id_test_num * 100
                    
                    id_err_num = id_false_reject_num + id_class_err_num
                    total_id_err_num += id_err_num
                    id_err_rate = id_err_num / id_test_num * 100
                else:
                    id_false_reject_rate = 0.0
                    id_err_rate = 0.0


                # if torch.any(id_mask):
                #     id_true_labels = true_labels[id_mask]
                #     id_pred_labels = results['prediction'][id_mask]
                #     id_err_num = torch.count_nonzero(id_pred_labels - id_true_labels)
                #     id_test_num = len(id_true_labels)
                #     id_err_rate = id_err_num / id_test_num * 100

                #     total_id_err_num += id_err_num
                #     total_id_test_num += id_test_num
                # else:
                #     id_err_rate = 0.0

                # 4. 计算 OOD 检测错误率
                if torch.any(ood_mask):
                    ood_confidence = confidence_scores[ood_mask]
                    # 错误：模型预测 OOD 样本为 ID（置信度 >= 0.4）
                    ood_err_num = np.sum(ood_confidence >= ood_threshold).item()
                    ood_test_num = len(ood_confidence)
                    ood_err_rate = ood_err_num / ood_test_num * 100

                    total_ood_err_num += ood_err_num
                    total_ood_test_num += ood_test_num
                else:
                    ood_err_rate = 0.0

                # 5. 计算EER
                # 获取相似度矩阵和真实标签
                # 为EER计算准备数据：将多分类问题转换为二分类问题
                eer_scores = []
                eer_labels = []
                # ===== 新增：OOD指标累积 =====
                total_ood_auroc = []
                total_ood_fpr95 = []
                total_ood_acc = []
                
                for i in range(len(true_labels)):
                    true_speaker = true_labels[i].item()
                    
                    # 跳过OOD样本
                    if true_speaker < 0:
                        continue
                        
                    # 确保索引在范围内
                    if i >= similarity_matrix.shape[0]:
                        print(f"Warning: Skipping sample {i} - index out of bounds")
                        continue
                        
                    try:
                        # 检查speaker索引是否有效
                        if true_speaker >= similarity_matrix.shape[1]:
                            print(f"Warning: Speaker {true_speaker} exceeds matrix columns {similarity_matrix.shape[1]}")
                            continue
                            
                        # 正例得分：与真实speaker原型的相似度
                        positive_score = similarity_matrix[i, true_speaker].item()
                        eer_scores.append(positive_score)
                        eer_labels.append(1)
                        
                        # 负例得分：与其他speaker原型的最大相似度
                        other_speakers = [j for j in range(similarity_matrix.shape[1]) if j != true_speaker]
                        if other_speakers:
                            negative_scores = similarity_matrix[i, other_speakers]
                            max_negative_score = torch.max(negative_scores).item()
                            eer_scores.append(max_negative_score)
                            eer_labels.append(0)

                    except Exception as e:
                        print(f"Error processing sample {i}: {e}")
                        continue

                # 计算EER
                if len(eer_scores) > 0:
                    eer, eer_threshold = calculate_eer(np.array(eer_scores), np.array(eer_labels))
                    total_eers.append(eer)
                else:
                    eer = 0.0
                    eer_threshold = 0.0
                    total_eers.append(eer)

                # ===== OOD检测评估 =====
                # 使用最大相似度作为置信度
                confidence_scores = torch.max(similarity_matrix, dim=1)[0].numpy()
                ood_labels = (true_labels >= 0).int().numpy()  # 1=in-domain, 0=OOD
                
                # 计算OOD指标
                if len(np.unique(ood_labels)) > 1:  # 确保有in-domain和OOD样本
                    # AUROC
                    auroc = roc_auc_score(ood_labels, confidence_scores)
                    
                    # FPR@95
                    fpr, tpr, thresholds = roc_curve(ood_labels, confidence_scores)
                    idx_95 = np.where(tpr >= 0.95)[0]
                    fpr_at_95 = fpr[idx_95[0]] if len(idx_95) > 0 else 1.0
                    
                    # 检测模型在最佳阈值下的理论最佳性能
                    best_acc = 0
                    best_thresh = 0
                    for thresh in thresholds:
                        preds = (confidence_scores >= thresh).astype(int)
                        acc = np.mean(preds == ood_labels)
                        if acc > best_acc:
                            best_acc = acc
                            best_thresh = thresh
                    
                    total_ood_auroc.append(auroc)
                    total_ood_fpr95.append(fpr_at_95)
                    total_ood_acc.append(best_acc)

                    f.write(f"针对 {family.name} - ID内部错误率: {id_class_err_rate:.2f}%, ID错分为OOD错误率: {id_false_reject_rate:.2f}%, OOD错分为ID错误率: {ood_err_rate:.2f}% - EER: {eer*100:.2f}% | OOD AUROC: {auroc:.4f}, FPR@95: {fpr_at_95:.4f}, Acc: {best_acc:.4f}\n")
                else:
                    f.write(f"针对 {family.name} - EER: {eer*100:.2f}% | OOD评估：数据不足\n")
                
                f.flush()

            # 最终统计
            # total_id_err_rate = total_id_err_num / total_id_test_num * 100
            total_id_class_err_rate = total_id_class_err_num / total_id_test_num * 100
            total_id_false_reject_rate = total_id_false_reject_num / total_id_test_num * 100
            total_ood_err_rate = total_ood_err_num / total_ood_test_num * 100

            total_err_ration = (total_id_class_err_num + total_id_false_reject_num + total_ood_err_num) / (total_id_test_num + total_ood_test_num) * 100

            mean_eer = np.mean(total_eers)
            mean_auroc = np.mean(total_ood_auroc)
            mean_fpr95 = np.mean(total_ood_fpr95)
            mean_acc = np.mean(total_ood_acc)
            
            f.write(f"\n=============================================\n")
            f.write(f"ID内部错误率: {total_id_class_err_rate:.2f}%\n")   # 模型错将 ID 样本预测为其他 ID
            f.write(f"ID错分为OOD错误率: {total_id_false_reject_rate:.2f}%\n")   # 模型错将 ID 样本预测为 OOD
            f.write(f"OOD错分为ID错误率: {total_ood_err_rate:.2f}%\n") # 模型错将 OOD 样本预测为 ID
            f.write(f"总错误率: {total_err_ration:.2f}%\n")
            f.write(f"In-domain平均EER: {mean_eer*100:.2f}%\n")
            f.write(f"OOD检测平均AUROC: {mean_auroc:.4f}\n")  # 模型区分 ID 和 OOD 样本的能力
            f.write(f"OOD检测平均FPR@95: {mean_fpr95:.4f}\n") # 如果要确保95%的in-domain样本被正确识别，需要承受多少OOD误判
            f.write(f"OOD检测平均准确率: {mean_acc:.4f}\n")   # 模型在最佳阈值下的理论最佳性能的平均值, 但是阈值在实际应用中不可知
        print("所有结果已保存到 txt 文件中!")
