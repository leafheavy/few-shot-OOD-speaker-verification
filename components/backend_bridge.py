"""
backend_bridge.py — 与用户现有代码的兼容层
将 few_shot_learning.py / baseline.py / dataloader.py 等统一封装为前端可调用的 API

所有 torch / 用户模块均延迟导入，保证在没有 GPU 的机器上前端仍能正常启动。
"""

from __future__ import annotations

import sys
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── torch 延迟导入 ──────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None   # type: ignore
    F     = None   # type: ignore


# ════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════

def _ensure_path(directory: str) -> None:
    """把目录加入 sys.path（幂等）"""
    resolved = str(Path(directory).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _inject_calculate_eer_shim(directory: str) -> None:
    """
    如果目录里没有 calculate_eer.py，自动写入一份最简实现，
    防止 few_shot_learning.py 等模块 import 时抛 ImportError。
    已有真实文件时直接跳过，不覆盖。
    """
    target = Path(directory) / "calculate_eer.py"
    if target.exists():
        return
    shim = (
        "import numpy as np\n\n"
        "def calculate_eer(scores, labels):\n"
        "    scores = np.asarray(scores, dtype=float)\n"
        "    labels = np.asarray(labels, dtype=int)\n"
        "    thresholds = np.sort(np.unique(scores))[::-1]\n"
        "    n_pos, n_neg = np.sum(labels==1), np.sum(labels==0)\n"
        "    if n_pos==0 or n_neg==0: return 0.0, 0.0\n"
        "    frrs, fars = [], []\n"
        "    for t in thresholds:\n"
        "        p = scores >= t\n"
        "        tp = np.sum(p & (labels==1)); fp = np.sum(p & (labels==0))\n"
        "        fn = np.sum(~p & (labels==1)); tn = np.sum(~p & (labels==0))\n"
        "        frrs.append(fn / max(fn+tp, 1)); fars.append(fp / max(fp+tn, 1))\n"
        "    frrs, fars = np.array(frrs), np.array(fars)\n"
        "    idx = np.argmin(np.abs(frrs - fars))\n"
        "    return float((frrs[idx]+fars[idx])/2), float(thresholds[idx])\n"
    )
    target.write_text(shim, encoding="utf-8")


# ════════════════════════════════════════════════
# 核心公开函数：导入用户模块
# ════════════════════════════════════════════════

def import_user_modules(user_code_dir: str) -> Dict[str, object]:
    """
    将 user_code_dir 加入 sys.path，然后逐一 import 用户后端脚本。

    Args:
        user_code_dir: 含有 few_shot_learning.py、baseline.py 等文件的目录。

    Returns:
        dict，键为模块名，值为已导入的模块对象；导入失败则值为 None。
    """
    _ensure_path(user_code_dir)
    _inject_calculate_eer_shim(user_code_dir)

    standard_modules = [
        "dataloader",
        "dataloader_OOD",
        "baseline",
        "baseline_OOD",
        "few_shot_learning",
    ]
    optional_modules = [
        "few_shot_learning_OOD_other_loss_xz",   # 依赖 loss_functions.py
    ]

    result: Dict[str, object] = {}

    for name in standard_modules:
        try:
            if name in sys.modules:
                mod = importlib.reload(sys.modules[name])
            else:
                mod = importlib.import_module(name)
            result[name] = mod
        except Exception as exc:
            result[name] = None
            print(f"[bridge] ⚠  无法导入 '{name}': {exc}")

    for name in optional_modules:
        try:
            if name in sys.modules:
                mod = importlib.reload(sys.modules[name])
            else:
                mod = importlib.import_module(name)
            result[name] = mod
        except Exception as exc:
            result[name] = None
            print(f"[bridge] ⚠  可选模块 '{name}' 不可用: {exc}")

    return result


# ════════════════════════════════════════════════
# DataLoader 工厂
# ════════════════════════════════════════════════

def create_family_loaders(
    dataset_root: str,
    modules: Dict,
    mode: str = "standard",
    batch_size: int = 8,
    preload: bool = True,
) -> Tuple:
    """
    根据运行模式选择对应的 dataloader 模块并创建 DataLoader。

    Returns:
        (support_loaders, test_loaders)  两个以 family 名为键的字典
    """
    root = Path(dataset_root)

    if mode in ("ood", "baseline_ood"):
        mod = modules.get("dataloader_OOD") or modules.get("dataloader")
    else:
        mod = modules.get("dataloader")

    if mod is None:
        raise ImportError(
            "dataloader 模块未能加载。\n"
            "请确认侧边栏「用户代码目录」指向包含 dataloader.py 的文件夹，"
            "并点击「加载用户代码模块」。"
        )

    return mod.create_dataloaders_for_families(
        root, batch_size=batch_size, preload=preload
    )


# ════════════════════════════════════════════════
# 少样本学习运行器
# ════════════════════════════════════════════════

class FewShotRunner:
    """
    统一封装 4 种模式：
      'baseline'  — InnerProductCalculator（余弦相似度，无梯度更新）
      'baseline_ood' — Baseline OOD 版（余弦相似度 + 开放集检测）
      'standard'  — FewShotLearning（原型网络 + 熵正则化）
      'ood'       — FewShotLearning OOD 版（ArcFace/CosFace + 开放集检测）
    """

    def __init__(self, modules: Dict, mode: str = "standard"):
        self.modules = modules
        self.mode    = mode
        self._model  = None
        self._init_model()

    def _init_model(self) -> None:
        if self.mode == "baseline":
            mod = self.modules.get("baseline")
            if mod:
                self._model = mod.InnerProductCalculator(metric="cosine")

        elif self.mode == "baseline_ood":
            mod = self.modules.get("baseline_OOD")
            if mod:
                self._model = mod.InnerProductCalculator(metric="cosine")
    
        elif self.mode == "standard":
            mod = self.modules.get("few_shot_learning")
            if mod:
                self._model = mod.FewShotLearning()

        elif self.mode == "ood":
            mod = self.modules.get("few_shot_learning_OOD_other_loss_xz")
            if mod:
                self._model = mod.FewShotLearning()
            else:
                # 回退到 standard
                mod = self.modules.get("few_shot_learning")
                if mod:
                    self._model = mod.FewShotLearning()
                    self.mode = "standard"

    def is_available(self) -> bool:
        return self._model is not None

    # ── 核心评估函数 ────────────────────────────

    def run_family(
        self,
        support_loader,
        test_loader,
        epochs: int = 100,
        lr: float = 1e-3,
        ood_threshold: float = 0.4,
    ) -> Dict:
        """
        对单个 family 执行「初始化 → 学习 → 推理 → 指标计算」完整流程。

        Returns:
            包含 err_rate / eer / similarity_matrix / true_labels /
            prediction 等字段的字典；OOD 模式额外含 auroc / fpr_at_95 等。
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 未安装，无法运行模型评估。")

        from utils.metrics import calculate_eer, calculate_ood_metrics

        m = self._model

        # 1. 初始化原型
        if self.mode in ("baseline", "baseline_ood"):
            m.build_W_matrix(support_loader)
        else:
            m.initialization(support_loader)

        # 2. 梯度更新（baseline 跳过）
        loss_curve = []
        if self.mode not in ("baseline", "baseline_ood"):
            ret = m.learning(support_loader, Epochs=epochs, lr=lr)
            if isinstance(ret, list):
                loss_curve = ret

        # 3. 推理
        if self.mode == "baseline_ood":
            raw = m.compute_inner_products(test_loader, threshold=ood_threshold)
        elif self.mode == "baseline":
            raw = m.compute_inner_products(test_loader)
        else:
            raw = m.calculate_result(test_loader)

        sim_mat     = raw["similarity_matrix"]   # Tensor [N, C]
        true_labels = raw["true_labels"]          # Tensor [N]
        pred        = raw["prediction"]           # Tensor [N]

        # 4. 分类错误率（仅计 in-domain）
        id_mask = true_labels >= 0
        if torch.any(id_mask):
            id_true = true_labels[id_mask]
            id_pred = pred[id_mask]
            test_num = int(id_mask.sum().item())

            if self.mode in ("ood", "baseline_ood"):
                id_reject_mask = id_pred < 0
                id_accept_mask = ~id_reject_mask

                id_false_reject_num = int(torch.count_nonzero(id_reject_mask).item())
                if torch.any(id_accept_mask):
                    id_class_err_num = int(torch.count_nonzero(id_pred[id_accept_mask] - id_true[id_accept_mask]).item())
                else:
                    id_class_err_num = 0

                err_num = id_false_reject_num + id_class_err_num
                err_rate = err_num / test_num * 100
            else:
                err_num = int(torch.count_nonzero(id_pred - id_true).item())
                err_rate = err_num / test_num * 100
        else:
            err_num = test_num = 0
            err_rate = 0.0
            id_false_reject_num = 0
            id_class_err_num = 0

        # 5. EER
        eer_scores: List[float] = []
        eer_labs:   List[int]   = []
        for i_t in torch.where(id_mask)[0]:
            i  = int(i_t.item())
            ts = int(true_labels[i].item())
            if ts >= sim_mat.shape[1]:
                continue
            eer_scores.append(float(sim_mat[i, ts].item()))
            eer_labs.append(1)
            others = [j for j in range(sim_mat.shape[1]) if j != ts]
            if others:
                eer_scores.append(float(torch.max(sim_mat[i, torch.tensor(others)]).item()))
                eer_labs.append(0)

        if eer_scores:
            eer, eer_thr = calculate_eer(np.array(eer_scores), np.array(eer_labs))
        else:
            eer, eer_thr = 0.0, 0.0

        result = {
            "err_rate"         : err_rate,
            "err_num"          : err_num,
            "test_num"         : test_num,
            "eer"              : eer,
            "eer_threshold"    : eer_thr,
            "similarity_matrix": sim_mat,
            "true_labels"      : true_labels,
            "prediction"       : pred,
        }
        if loss_curve:
            result["loss_curve"] = loss_curve
            
        # 6. OOD 指标
        if self.mode in ("ood", "baseline_ood"):
            ood_mask     = ~id_mask
            confidence   = torch.max(sim_mat, dim=1)[0].detach().numpy()
            ood_label_np = id_mask.numpy().astype(int)

            if torch.any(ood_mask):
                ood_m = calculate_ood_metrics(confidence, ood_label_np)

                id_conf = confidence[id_mask.numpy()]
                id_fr   = int(np.sum(id_conf < ood_threshold))

                ood_conf     = confidence[ood_mask.numpy()]
                ood_err_num  = int(np.sum(ood_conf >= ood_threshold))
                ood_test_num = int(ood_mask.sum())

                result.update({
                    "id_false_reject_rate": id_fr / max(int(id_mask.sum()), 1) * 100,
                    "id_class_err_rate"   : id_class_err_num / max(int(id_mask.sum()), 1) * 100,
                    "ood_err_rate"        : ood_err_num / max(ood_test_num, 1) * 100,
                    **ood_m,
                })

        return result

    def export_classifier_state(self, support_loader, family_name: Optional[str] = None) -> Dict[str, Any]:
        """
        导出当前 family 训练后的分类器状态，便于缓存与后续单样本推理。
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 未安装，无法导出分类器状态。")

        if self._model is None:
            raise RuntimeError("模型未初始化，无法导出分类器状态。")

        dataset = getattr(support_loader, "dataset", None)
        speaker_to_label = getattr(dataset, "speaker_to_label", {}) if dataset is not None else {}
        label_to_speaker = {int(v): str(k) for k, v in speaker_to_label.items()}

        state: Dict[str, Any] = {
            "family_name": family_name or "",
            "mode": self.mode,
            "label_to_speaker": label_to_speaker,
        }

        m = self._model
        if self.mode in ("baseline", "baseline_ood"):
            if getattr(m, "W", None) is None or getattr(m, "support_labels", None) is None:
                raise RuntimeError("baseline 分类器状态为空，请先完成 family 训练/构建。")
            state.update({
                "weight": m.W.detach().cpu().numpy(),
                "support_labels": m.support_labels.detach().cpu().numpy(),
                "use_bias": False,
            })
        else:
            weight = None
            bias = None
            if getattr(m, "classifier", None) is not None:
                weight = m.classifier.weight.detach().cpu().numpy()
                bias = m.classifier.bias.detach().cpu().numpy()
            elif getattr(m, "W", None) is not None:
                weight = m.W.detach().cpu().numpy()
                b = getattr(m, "b", None)
                if b is not None:
                    bias = b.detach().cpu().numpy() if hasattr(b, "detach") else np.asarray(b, dtype=np.float32)

            if weight is None:
                raise RuntimeError("few-shot 分类器状态为空，请先完成 family 训练。")

            state.update({
                "weight": np.asarray(weight, dtype=np.float32),
                "bias": np.asarray(bias, dtype=np.float32) if bias is not None else None,
                "use_bias": bias is not None,
            })

        return state


def predict_embedding_with_state(
    embedding: np.ndarray,
    classifier_state: Dict[str, Any],
    ood_threshold: float = 0.4,
) -> Dict[str, Any]:
    """
    使用导出的分类器状态对单条 embedding 进行预测。
    """
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    w = np.asarray(classifier_state["weight"], dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"分类器权重维度非法: {w.shape}")
    if vec.shape[0] != w.shape[1]:
        raise ValueError(f"embedding 维度({vec.shape[0]})与分类器维度({w.shape[1]})不匹配")

    vec_norm = vec / max(np.linalg.norm(vec), 1e-12)
    mode = classifier_state.get("mode", "standard")
    use_bias = bool(classifier_state.get("use_bias", False))

    if mode in ("baseline", "baseline_ood"):
        scores = np.dot(w, vec_norm)  # [num_support]
        support_labels = np.asarray(classifier_state.get("support_labels"))
        if support_labels.size == 0:
            raise ValueError("baseline 分类器缺少 support_labels。")
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        pred_label = int(support_labels[best_idx])
    else:
        logits = np.dot(w, vec_norm)  # [num_classes]
        if use_bias and classifier_state.get("bias") is not None:
            logits = logits + np.asarray(classifier_state["bias"], dtype=np.float32)
        best_idx = int(np.argmax(logits))
        best_score = float(logits[best_idx])
        pred_label = best_idx

    if mode in ("baseline_ood", "ood") and best_score < ood_threshold:
        pred_label = -1

    label_to_speaker = classifier_state.get("label_to_speaker", {}) or {}
    pred_speaker = "OOD/未知" if pred_label < 0 else label_to_speaker.get(pred_label, f"label_{pred_label}")

    return {
        "family_name": classifier_state.get("family_name", ""),
        "mode": mode,
        "pred_label": pred_label,
        "pred_speaker_id": pred_speaker,
        "confidence": best_score,
        "ood_threshold": ood_threshold,
    }

# ════════════════════════════════════════════════
# 嵌入收集（UMAP 可视化辅助）
# ════════════════════════════════════════════════

def collect_embeddings_from_loader(loader) -> Tuple[np.ndarray, np.ndarray]:
    """从 DataLoader 收集全部嵌入向量和标签，返回 numpy 数组。"""
    all_embs, all_labels = [], []
    for embs, labels, _, _ in loader:
        all_embs.append(embs.numpy())
        all_labels.append(labels.numpy())
    return np.vstack(all_embs), np.concatenate(all_labels)