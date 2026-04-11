# 指标计算文档

本文档汇总了代码库中所有指标的定义和计算方法。

---

## 1. 说话人验证指标

### 1.1 EER (Equal Error Rate) - 等错误率

EER 是 FAR（错误接受率）和 FRR（错误拒绝率）相等时的错误率，是说话人验证系统的核心指标。

#### 方法一：基于 ROC 曲线（推荐）

**文件**: `calculate_eer.py`

```python
def calculate_eer(scores: np.ndarray, labels: np.ndarray,
                  plot_curve: bool = False, title_suffix: str = "") -> Tuple[float, float]:
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # 计算假负率(FNR) = 1 - TPR
    fnr = 1 - tpr

    # 找到FPR和FNR最接近的点，即EER点
    diff = np.abs(fpr - fnr)
    eer_index = np.argmin(diff)

    eer = fpr[eer_index]
    eer_threshold = thresholds[eer_index]

    return eer, eer_threshold
```

#### 方法二：手动实现（均匀采样阈值）

```python
def calculate_eer_manual(scores: np.ndarray, labels: np.ndarray,
                         num_thresholds: int = 1000) -> Tuple[float, float]:
    # 生成均匀分布的阈值
    thresholds = np.linspace(np.min(scores), np.max(scores), num_thresholds)

    far_list, frr_list = [], []

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)

        # FAR = FP / (FP + TN)
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        far = fp / (fp + tn) if (fp + tn) > 0 else 0

        # FRR = FN / (FN + TP)
        fn = np.sum((predictions == 0) & (labels == 1))
        tp = np.sum((predictions == 1) & (labels == 1))
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0

        far_list.append(far)
        frr_list.append(frr)

    # 找到FAR和FRR差值最小的点
    diff = np.abs(np.array(far_list) - np.array(frr_list))
    eer_index = np.argmin(diff)

    eer = far_list[eer_index]
    eer_threshold = thresholds[eer_index]

    return eer, eer_threshold
```

#### 方法三：NIST 官方实现

**文件**: `speakerlab/utils/score_metrics.py`

```python
def compute_eer(fnr, fpr, scores=None):
    """ computes the equal error rate (EER) given FNR and FPR values calculated
        for a range of operating points on the DET curve
    """
    diff_pm_fa = fnr - fpr
    x1 = np.flatnonzero(diff_pm_fa >= 0)[0]
    x2 = np.flatnonzero(diff_pm_fa < 0)[-1]
    a = (fnr[x1] - fpr[x1]) / (fpr[x2] - fpr[x1] - (fnr[x2] - fnr[x1]))

    if scores is not None:
        score_sort = np.sort(scores)
        return fnr[x1] + a * (fnr[x2] - fnr[x1]), score_sort[x1]

    return fnr[x1] + a * (fnr[x2] - fnr[x1])
```

---

### 1.2 FAR / FRR

- **FAR (False Acceptance Rate, FPR)**: 错误接受率，冒名顶替者被错误接受的比率
  - `FAR = FP / (FP + TN)`

- **FRR (False Rejection Rate, FNR)**: 错误拒绝率，真正的目标说话人被错误拒绝的比率
  - `FRR = FN / (FN + TP)`

---

### 1.3 DCF (Detection Cost Function)

**文件**: `speakerlab/utils/score_metrics.py`

检测成本函数，衡量系统的检测代价。

```python
def compute_c_norm(fnr, fpr, p_target, c_miss=1, c_fa=1):
    """ computes normalized minimum detection cost function (DCF)
        given the costs for false accepts and false rejects as well as
        a priori probability for target speakers
    """
    c_det = min(c_miss * fnr * p_target + c_fa * fpr * (1 - p_target))
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))

    return c_det / c_def
```

参数说明：
- `p_target`: 目标说话人的先验概率
- `c_miss`: 漏检的代价（FN）
- `c_fa`: 误检的代价（FP）

---

## 2. OOD 检测指标

### 2.1 AUROC (Area Under ROC Curve)

**文件**: `utils/metrics.py`

ROC 曲线下面积，用于衡量 OOD 检测器的整体性能。

```python
from sklearn.metrics import roc_auc_score

auroc = float(roc_auc_score(ood_labels, confidence_scores))
```

### 2.2 FPR@95

在 TPR（真正率）达到 95% 时的假正率（FPR）。衡量高召回率下的误检情况。

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(ood_labels, confidence_scores)
idx_95 = np.where(tpr >= 0.95)[0]
fpr_at_95 = float(fpr[idx_95[0]]) if len(idx_95) > 0 else 1.0
```

### 2.3 最佳准确率 (Best Accuracy)

在最佳阈值下的二分类准确率。

```python
best_acc, best_threshold = 0.0, 0.0
for thresh in thresholds:
    preds = (confidence_scores >= thresh).astype(int)
    acc = float(np.mean(preds == ood_labels))
    if acc > best_acc:
        best_acc = acc
        best_threshold = float(thresh)
```

### 2.4 完整 OOD 指标计算

**文件**: `utils/metrics.py`

```python
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
```

---

## 3. 说话人分割指标

### 3.1 DER (Diarization Error Rate)

**文件**: `egs/3dspeaker/speaker-diarization/local/DER.py`

DER 由三部分组成：
- **Missed Speech (MS)**: 漏检的语音时间
- **False Alarm (FA)**: 误检的语音时间
- **Speaker Error Rate (SER)**: 说话人混淆错误

```
DER = MS + FA + SER
```

---

## 4. 分类指标

### 4.1 Accuracy

**文件**: `speakerlab/utils/utils.py`

```python
def accuracy(x, target):
    # x: [*, C], target: [*,]
    _, pred = x.topk(1)
    pred = pred.squeeze(-1)
    acc = pred.eq(target).float().mean()
    return acc * 100
```

### 4.2 Precision / Recall / F1

**文件**: `egs/3dspeaker/language-identification/local/compute_acc.py`

```python
# Accuracy
accuracy = cor / sum * 100

# Recall
recall = data[i, i] / (float(form[i, num]) + 1e-10)

# Precision
precision = data[i, i] / (float(form[num, i]) + 1e-10)

# F1
f1 = 2 * precision * recall / (precision + recall + 1e-10)
```

---

### 4.3 Average Precision (mAP)

**文件**: `speakerlab/utils/utils.py`

```python
def average_precision(scores, labels):
    """
    计算平均精度
    scores: [N, ], labels: [N, ]
    """
    sort_idx = np.argsort(scores)[::-1]
    scores = scores[sort_idx]
    labels = labels[sort_idx]
    tp_count = (labels == 1).sum()
    tp = labels.cumsum()
    recall = tp / tp_count
    precision = tp / (np.arange(len(labels)) + 1)

    # Smooth precision
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    # 计算AP
    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    average_precision = np.sum(
        (recall[indices] - recall[indices - 1]) * precision[indices])
    return average_precision
```

---

## 5. 结果汇总

### 5.1 Family 结果聚合

**文件**: `utils/metrics.py`

```python
def aggregate_family_results(family_results: list) -> Dict:
    """
    汇总所有 family 的评估结果
    """
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
```

---

## 6. 指标汇总表

| 指标类别 | 指标名称 | 文件位置 | 用途 |
|---------|---------|---------|------|
| **验证指标** | EER | `calculate_eer.py`, `utils/metrics.py`, `speakerlab/utils/score_metrics.py` | 说话人验证核心指标 |
| | FAR/FPR | `calculate_eer.py` | 安全场景评估 |
| | FRR/FNR | `calculate_eer.py` | 用户体验评估 |
| | minDCF | `speakerlab/utils/score_metrics.py` | 检测成本函数 |
| **OOD检测** | AUROC | `utils/metrics.py` | OOD检测能力 |
| | FPR@95 | `utils/metrics.py` | 高召回下的误检率 |
| | 最佳准确率 | `utils/metrics.py` | 理论最佳性能 |
| **分类指标** | Accuracy | `speakerlab/utils/utils.py` | 分类正确率 |
| | Precision/Recall/F1 | `egs/3dspeaker/language-identification/local/compute_acc.py` | 语言识别 |
| | mAP | `speakerlab/utils/utils.py` | 平均精度 |
| **分割指标** | DER | `egs/3dspeaker/speaker-diarization/local/DER.py` | 说话人分割错误率 |
| | MS/FA/SER | `egs/3dspeaker/speaker-diarization/local/DER.py` | DER组成部分 |

---

## 7. 代码文件索引

| 文件路径 | 描述 |
|---------|------|
| `calculate_eer.py` | 主EER计算模块 |
| `utils/metrics.py` | 项目核心指标模块 |
| `speakerlab/utils/score_metrics.py` | NIST SRE官方指标 |
| `speakerlab/utils/utils.py` | 辅助指标函数 |
| `egs/3dspeaker/speaker-diarization/local/DER.py` | 说话人分割DER |
| `egs/3dspeaker/language-identification/local/compute_acc.py` | 分类指标 |