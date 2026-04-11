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
    def __init__(self):
        """
        小样本学习器
        
        Args:
            
        """
        self.classifier = None
        self.W = None  # 支撑集嵌入矩阵 [num_support, embedding_dim]
        self.b = None # 偏置项
        self.support_labels = None
        self.support_speaker_ids = None
        self.num_classes = 0 # 支持集中的类别数
    
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
                # print(f"Batch embeddings shape: {embeddings.shape}")  # 添加此行：检查每个batch的嵌入形状
                all_embeddings.append(embeddings)
                all_labels.append(labels)
                all_speaker_ids.extend(speaker_ids)
        
        self.support_labels = torch.cat(all_labels, dim=0)
        unique_labels = torch.unique(self.support_labels)
        self.num_classes = len(unique_labels)
        
        # 拼接所有支撑集的嵌入
        self.W = torch.cat(all_embeddings, dim=0)  # [num_support, embedding_dim]

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
        self.W = F.normalize(prototypes_tensor, p=2, dim=1)

        self.b = torch.zeros(self.num_classes)  # 动态初始化偏置项

        self.support_speaker_ids = all_speaker_ids
        
        # logger.info(f"构建嵌入矩阵 W: {self.W.shape}, samples of the support set: {len(self.support_labels)}")
        # return self.W
        return

    def learning(self, support_loader, Epochs=100, lr=0.001):
        if self.W is None:
            raise ValueError("必须先调用 initialization() 方法构建W矩阵")

        self.classifier = CosineClassifier(self.W, self.b)
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.classifier.train()

        loss_curve = []
        for epoch in range(Epochs):
            epoch_loss = 0.0
            for batch in support_loader:
                embeddings, labels, speaker_ids, _ = batch
                
                optimizer.zero_grad()

                outputs = self.classifier(embeddings)

                loss = criterion(outputs, labels)  # 交叉熵损失
                entropy_reg = softmax_entropy(outputs).mean()  # 熵正则化
                total_loss = loss + 0.01 * entropy_reg  # 总损失
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            avg_loss = epoch_loss / len(support_loader)
            loss_curve.append(float(avg_loss))
            # if (epoch + 1) % 10 == 0 or epoch == 0:
                # logger.info(f"Epoch [{epoch+1}/{Epochs}], Loss: {avg_loss:.4f}")
        # logger.info(f"Epoch [{epoch+1}/{Epochs}], Loss: {avg_loss:.4f}")
        
        # 更新后的 W 和 b
        self.W = self.classifier.weight.data
        self.b = self.classifier.bias.data

        return loss_curve

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
        
        with torch.no_grad():
            for batch in test_loader:
                embeddings, labels, speaker_ids, _ = batch

                outputs = self.classifier(embeddings)
                predicted_labels = torch.argmax(outputs, dim=1)

                all_test_embeddings.append(embeddings) # 已由 pretrained model 提取得到的 embedding
                all_true_labels.append(labels) # 0~4 五个 speaker id 的映射
                all_test_speaker_ids.extend(speaker_ids) # 五个 speaker id
                all_pred_labels.append(predicted_labels) # 预测的标签 0~4
        
        # 计算相似度矩阵用于EER计算
        X_test = torch.cat(all_test_embeddings, dim=0)  # [num_test, embedding_dim]
        # 对测试嵌入进行归一化
        X_test_normalized = F.normalize(X_test, p=2, dim=1)
        similarity_matrix = torch.mm(X_test_normalized, self.W.t()) # [num_test, num_speaker]

        true_labels = torch.cat(all_true_labels, dim=0)
        predictions = torch.cat(all_pred_labels, dim=0)

        return {
            'similarity_matrix': similarity_matrix,
            'test_embeddings': X_test, 
            'true_labels': true_labels,
            'test_speaker_ids': all_test_speaker_ids,
            'W_matrix': self.W,
            'support_labels': self.support_labels,
            'support_speaker_ids': self.support_speaker_ids,
            'prediction': predictions,  # 现在范围是 0~4
            'true_labels': true_labels,       # 范围也是 0~4
            'test_speaker_ids': all_test_speaker_ids
        }

if __name__ == "__main__":
    with open("few_shot_learning_result1.txt", "w") as f:
        # 初始化内积计算器
        mymodel = FewShotLearning()
        
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
            mymodel.initialization(support_loader) # shape: (num_support, embedding_dim)
            
            # 2. 小样本学习
            mymodel.learning(support_loader, Epochs=100, lr=0.001)

            # 3. 测试
            results = mymodel.calculate_result(test_loader)
            
            pred = results['prediction'] # shape: (num_test,)
            true_labels = results['true_labels'] # shape: (num_test,)

            # 4. 统计错判的个数
            err_num = torch.count_nonzero(pred - true_labels)
            test_num = len(true_labels)

            total_err_num += err_num
            total_test_num += test_num

            err_ration = err_num / test_num * 100

            # 5. 计算EER
            # 获取相似度矩阵和真实标签
            similarity_matrix = results['similarity_matrix']  # [num_test, num_speaker]
            
            # 为EER计算准备数据：将多分类问题转换为二分类问题
            eer_scores = []
            eer_labels = []
            
            for i in range(len(true_labels)):
                true_speaker = true_labels[i].item()
                
                # 对于每个测试样本，找到其与真实说话人支持样本的最大相似度作为正例得分
                # 在Few-Shot Learning中，W矩阵的每一行对应一个类别的原型向量
                if true_speaker < similarity_matrix.shape[1]:  # 确保标签在有效范围内
                    positive_score = similarity_matrix[i, true_speaker].item()
                    eer_scores.append(positive_score)
                    eer_labels.append(1)  # 正例
                    
                    # 找到与其他说话人原型向量的最大相似度作为负例得分
                    # 排除真实类别，计算与其他所有类别的最大相似度
                    other_classes = [j for j in range(similarity_matrix.shape[1]) if j != true_speaker]
                    if other_classes:
                        negative_scores = similarity_matrix[i, other_classes]
                        max_negative_score = torch.max(negative_scores).item()
                        eer_scores.append(max_negative_score)
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
    print("所有结果已保存到 few_shot_learning_result1.txt")
