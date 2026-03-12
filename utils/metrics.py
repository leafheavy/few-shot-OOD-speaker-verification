"""
metrics.py — 评估指标计算
实现 EER、AUROC、FPR@95 等指标，无需外部 calculate_eer 模块依赖
"""
import numpy as np
from typing import Tuple, Dict, Optional


# ──────────────────────────────────────────────
# EER (Equal Error Rate)
# ──────────────────────────────────────────────

def calculate_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    计算等错误率 (EER)
    
    Args:
        scores: 相似度得分数组，正例得分高
        labels: 二值标签，1=正例，0=负例
    
    Returns:
        eer:       等错误率 [0, 1]
        threshold: 对应 EER 的阈值
    """
    # 对阈值从高到低排序
    thresholds = np.sort(np.unique(scores))[::-1]

    frr_list, far_list = [], []
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.0

    for thresh in thresholds:
        pred_pos = scores >= thresh
        tp = np.sum(pred_pos & (labels == 1))
        fp = np.sum(pred_pos & (labels == 0))
        fn = np.sum(~pred_pos & (labels == 1))
        tn = np.sum(~pred_pos & (labels == 0))

        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Rejection Rate
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Acceptance Rate
        frr_list.append(frr)
        far_list.append(far)

    frr_arr = np.array(frr_list)
    far_arr = np.array(far_list)

    # 找 FRR ≈ FAR 的交叉点
    diff = np.abs(frr_arr - far_arr)
    idx = np.argmin(diff)

    eer = (frr_arr[idx] + far_arr[idx]) / 2.0
    threshold = thresholds[idx]

    return float(eer), float(threshold)


# ──────────────────────────────────────────────
# OOD 检测指标
# ──────────────────────────────────────────────

def calculate_ood_metrics(
    confidence_scores: np.ndarray,
    ood_labels: np.ndarray,
) -> Dict[str, float]:
    """
    计算 OOD 检测指标
    
    Args:
        confidence_scores: 模型置信度分数 (越高越可能是 in-domain)
        ood_labels:        真实标签 1=in-domain, 0=OOD
    
    Returns:
        {
            'auroc':    AUROC 分数 [0,1]
            'fpr_at_95': FPR@TPR=95% [0,1]
            'best_acc': 最佳阈值下的二分类准确率 [0,1]
            'best_threshold': 对应最佳准确率的阈值
        }
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    if len(np.unique(ood_labels)) < 2:
        return {"auroc": 0.0, "fpr_at_95": 1.0, "best_acc": 0.5, "best_threshold": 0.0}

    # AUROC
    auroc = float(roc_auc_score(ood_labels, confidence_scores))

    # FPR@95
    fpr, tpr, thresholds = roc_curve(ood_labels, confidence_scores)
    idx_95 = np.where(tpr >= 0.95)[0]
    fpr_at_95 = float(fpr[idx_95[0]]) if len(idx_95) > 0 else 1.0

    # 最佳准确率
    best_acc, best_threshold = 0.0, 0.0
    for thresh in thresholds:
        preds = (confidence_scores >= thresh).astype(int)
        acc = float(np.mean(preds == ood_labels))
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(thresh)

    return {
        "auroc": auroc,
        "fpr_at_95": fpr_at_95,
        "best_acc": best_acc,
        "best_threshold": best_threshold,
    }


# ──────────────────────────────────────────────
# Family 结果汇总
# ──────────────────────────────────────────────

def aggregate_family_results(family_results: list) -> Dict:
    """
    汇总所有 family 的评估结果
    
    Args:
        family_results: 每个元素为单 family 的指标字典
    
    Returns:
        汇总指标字典
    """
    if not family_results:
        return {}

    total_err_num = sum(r.get("err_num", 0) for r in family_results)
    total_test_num = sum(r.get("test_num", 0) for r in family_results)
    eers = [r.get("eer", 0.0) for r in family_results if "eer" in r]

    result = {
        "total_err_rate": total_err_num / max(total_test_num, 1) * 100,
        "mean_eer": float(np.mean(eers)) * 100 if eers else 0.0,
        "n_families": len(family_results),
    }

    # OOD 指标 (若存在)
    if "auroc" in family_results[0]:
        result["mean_auroc"] = float(np.mean([r["auroc"] for r in family_results]))
        result["mean_fpr95"] = float(np.mean([r["fpr_at_95"] for r in family_results]))
        result["mean_ood_err_rate"] = float(np.mean(
            [r.get("ood_err_rate", 0) for r in family_results]
        ))
        result["mean_id_class_err_rate"] = float(np.mean(
            [r.get("id_class_err_rate", 0) for r in family_results]
        ))
        result["mean_id_false_reject_rate"] = float(np.mean(
            [r.get("id_false_reject_rate", 0) for r in family_results]
        ))

    return result