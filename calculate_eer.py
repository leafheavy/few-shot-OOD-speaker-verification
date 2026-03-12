import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from typing import Tuple, Union

def calculate_eer(scores: np.ndarray, labels: np.ndarray, 
                  plot_curve: bool = False, title_suffix: str = "") -> Tuple[float, float]:
    """
    计算等错误率(Equal Error Rate, EER)
    
    Args:
        scores: 相似度得分数组，形状为(n_samples,)
        labels: 真实标签数组，形状为(n_samples,)，0表示负例(imposter)，1表示正例(target)
        plot_curve: 是否绘制DET曲线和ROC曲线
        title_suffix: 图表标题的后缀，用于区分不同的family
    
    Returns:
        eer: 等错误率，取值范围[0, 1]
        eer_threshold: 对应的阈值
    """
    
    # 确保输入为numpy数组
    scores = np.array(scores)
    labels = np.array(labels)
    
    # 验证输入形状
    if scores.shape != labels.shape:
        raise ValueError(f" scores和labels形状不匹配: {scores.shape} vs {labels.shape}")
    
    # 计算ROC曲线 [1,3](@ref)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # 计算假负率(FNR) = 1 - TPR [1](@ref)
    fnr = 1 - tpr
    
    # 找到FPR和FNR最接近的点，即EER点 [1,3](@ref)
    # 使用绝对差值的最小值来定位EER
    diff = np.abs(fpr - fnr)
    eer_index = np.argmin(diff)
    
    # EER可以取FPR或FNR在该点的值（理论上应该相等）
    eer = fpr[eer_index]
    eer_threshold = thresholds[eer_index]
    
    # 可选：绘制曲线 [3](@ref)
    if plot_curve:
        _plot_eer_curves(fpr, tpr, fnr, thresholds, eer, eer_threshold, title_suffix)
    
    return eer, eer_threshold

def calculate_eer_manual(scores: np.ndarray, labels: np.ndarray, 
                         num_thresholds: int = 1000) -> Tuple[float, float]:
    """
    手动实现EER计算，通过均匀采样阈值 [2,4](@ref)
    
    Args:
        scores: 相似度得分数组
        labels: 真实标签数组
        num_thresholds: 阈值采样数量
    
    Returns:
        eer: 等错误率
        eer_threshold: 对应的阈值
    """
    scores = np.array(scores)
    labels = np.array(labels)
    
    # 生成均匀分布的阈值 [4](@ref)
    thresholds = np.linspace(np.min(scores), np.max(scores), num_thresholds)
    
    far_list = []  # 错误接受率(False Acceptance Rate)
    frr_list = []  # 错误拒绝率(False Rejection Rate)
    
    for threshold in thresholds:
        # 预测结果：得分 >= 阈值为正例，否则为负例
        predictions = (scores >= threshold).astype(int)
        
        # 计算FAR（负例被错误接受的比例） [2](@ref)
        # FAR = FP / (FP + TN)
        tn_mask = (labels == 0)
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # 计算FRR（正例被错误拒绝的比例） [2](@ref)
        # FRR = FN / (FN + TP)
        tp_mask = (labels == 1)
        fn = np.sum((predictions == 0) & (labels == 1))
        tp = np.sum((predictions == 1) & (labels == 1))
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        far_list.append(far)
        frr_list.append(frr)
    
    far_list = np.array(far_list)
    frr_list = np.array(frr_list)
    
    # 找到FAR和FRR差值最小的点 [3](@ref)
    diff = np.abs(far_list - frr_list)
    eer_index = np.argmin(diff)
    
    eer = far_list[eer_index]  # 或 frr_list[eer_index]，理论上相等
    eer_threshold = thresholds[eer_index]
    
    return eer, eer_threshold

def _plot_eer_curves(fpr: np.ndarray, tpr: np.ndarray, fnr: np.ndarray, 
                    thresholds: np.ndarray, eer: float, eer_threshold: float,
                    title_suffix: str = ""):
    """绘制ROC曲线和DET曲线"""
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC曲线 [3](@ref)
    ax1.plot(fpr, tpr, 'b-', linewidth=2, label='ROC Curve')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Guess')
    
    # 标记EER点
    eer_index = np.argmin(np.abs(fpr - (1 - tpr)))
    ax1.plot(fpr[eer_index], tpr[eer_index], 'ro', markersize=8, 
             label=f'EER Points ({eer:.3f})')

    ax1.set_xlabel('FPR')
    ax1.set_ylabel('TPR')
    ax1.set_title(f'ROC Curve {title_suffix}')
    ax1.legend()
    ax1.grid(True)
    
    # DET曲线 [3](@ref)
    ax2.plot(fpr, fnr, 'g-', linewidth=2, label='DET Curve')
    
    # 标记EER点
    ax2.plot(fpr[eer_index], fnr[eer_index], 'ro', markersize=8, 
             label=f'EER Points ({eer:.3f})')
    
    ax2.set_xlabel('FAR')
    ax2.set_ylabel('FRR')
    ax2.set_title(f'DET Curve {title_suffix}')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'eer_curve_{title_suffix.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 测试函数
if __name__ == "__main__":
    # 生成测试数据
    np.random.seed(42)
    
    # 正例（目标说话人）得分通常较高
    target_scores = np.random.normal(0.7, 0.15, 100)
    target_labels = np.ones(100)
    
    # 负例（冒名顶替者）得分通常较低
    imposter_scores = np.random.normal(0.3, 0.15, 100)
    imposter_labels = np.zeros(100)
    
    # 合并数据
    all_scores = np.concatenate([target_scores, imposter_scores])
    all_labels = np.concatenate([target_labels, imposter_labels])
    
    # 测试两种方法
    eer1, threshold1 = calculate_eer(all_scores, all_labels, plot_curve=True, title_suffix="Test Family")
    eer2, threshold2 = calculate_eer_manual(all_scores, all_labels)
    
    print(f"ROC-EER: {eer1:.4f}, Threshold: {threshold1:.4f}")
    print(f"Manual-EER: {eer2:.4f}, Threshold: {threshold2:.4f}")