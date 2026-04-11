"""
visualization.py — Streamlit 可视化组件
提供结果图表、UMAP嵌入图、相似度矩阵等可视化
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional


# ──────────────────────────────────────────────
# Family 级别结果折线/柱状图
# ──────────────────────────────────────────────

def plot_family_metrics(family_results: List[Dict], mode: str = "standard") -> go.Figure:
    """
    绘制各 family 的错误率和 EER 对比图
    
    Args:
        family_results: [{family_name, err_rate, eer, ...}, ...]
        mode: 'standard' | 'ood'
    """
    names = [r["family_name"] for r in family_results]
    err_rates = [r.get("err_rate", 0.0) for r in family_results]
    eers = [r.get("eer", 0.0) * 100 for r in family_results]

    if mode == "ood":
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["ID内部分类错误率(%)", "ID误拒率(%)", "OOD误接率(%)", "EER(%)"],
            vertical_spacing=0.18, horizontal_spacing=0.12,
        )
        metrics = [
            ("id_class_err_rate", 1, 1, "rgba(66,133,244,0.75)"),
            ("id_false_reject_rate", 1, 2, "rgba(234,67,53,0.75)"),
            ("ood_err_rate", 2, 1, "rgba(251,188,5,0.75)"),
            ("eer", 2, 2, "rgba(52,168,83,0.75)"),
        ]
        for key, row, col, color in metrics:
            vals = [r.get(key, 0.0) * (100 if key == "eer" else 1) for r in family_results]
            fig.add_trace(go.Bar(
                x=names, y=vals,
                marker_color=color, name=key,
                showlegend=False,
            ), row=row, col=col)
        fig.update_layout(height=540, title_text="各 Family OOD 评估指标")
    else:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["错误率 (Error Rate %)", "等错误率 (EER %)"],
            horizontal_spacing=0.12,
        )
        fig.add_trace(go.Bar(
            x=names, y=err_rates,
            marker_color="rgba(66,133,244,0.75)",
            name="错误率",
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=names, y=eers,
            marker_color="rgba(52,168,83,0.75)",
            name="EER",
        ), row=1, col=2)

        # 均值参考线
        mean_err = np.mean(err_rates)
        mean_eer = np.mean(eers)
        fig.add_hline(y=mean_err, line_dash="dash", line_color="red",
                      annotation_text=f"均值 {mean_err:.2f}%", row=1, col=1)
        fig.add_hline(y=mean_eer, line_dash="dash", line_color="green",
                      annotation_text=f"均值 {mean_eer:.2f}%", row=1, col=2)
        fig.update_layout(height=380, title_text="各 Family 评估指标")

    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
    fig.update_layout(
        plot_bgcolor="rgba(248,249,250,1)",
        paper_bgcolor="white",
        margin=dict(l=50, r=30, t=70, b=80),
        showlegend=False,
    )
    return fig


def plot_ood_radar(summary: Dict) -> go.Figure:
    """绘制 OOD 综合指标雷达图"""
    categories = [
        "AUROC", "1-FPR@95", "OOD检测准确率",
        "1-OOD误接率/100", "1-ID误拒率/100",
    ]
    values = [
        summary.get("mean_auroc", 0),
        1.0 - summary.get("mean_fpr95", 1.0),
        summary.get("mean_best_acc", 0),
        1.0 - summary.get("mean_ood_err_rate", 100) / 100,
        1.0 - summary.get("mean_id_false_reject_rate", 100) / 100,
    ]
    values = [max(0.0, min(1.0, v)) for v in values]
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(66,133,244,0.25)",
        line=dict(color="rgba(66,133,244,0.9)", width=2),
        name="OOD指标",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="OOD 检测综合能力雷达图",
        height=400,
        margin=dict(l=60, r=60, t=70, b=40),
    )
    return fig


# ──────────────────────────────────────────────
# UMAP 嵌入可视化
# ──────────────────────────────────────────────

def plot_umap(
    embeddings: np.ndarray,
    labels: List,
    title: str = "UMAP 说话人嵌入空间",
    n_neighbors: int = 10,
    min_dist: float = 0.3,
    ood_mask: Optional[np.ndarray] = None,
) -> Optional[go.Figure]:
    """
    使用 UMAP 降维并可视化说话人嵌入
    
    Args:
        embeddings:  (N, D) 嵌入矩阵
        labels:      N 个标签 (int 或 str)
        ood_mask:    N 个布尔值，True=OOD 样本 (显示为灰色X)
    """
    try:
        from umap import UMAP
    except ImportError:
        return None

    n = len(embeddings)
    if n < 4:
        return None

    actual_n_neighbors = min(n_neighbors, max(2, n - 1))
    reducer = UMAP(
        n_neighbors=actual_n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=42,
        n_jobs=1,
    )
    coords = reducer.fit_transform(embeddings)  # (N, 2)

    # 标签字符串化
    str_labels = [str(l) for l in labels]
    unique_labels = list(dict.fromkeys(str_labels))
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
    color_map = {lbl: palette[i % len(palette)] for i, lbl in enumerate(unique_labels)}

    fig = go.Figure()

    # in-domain 样本
    for lbl in unique_labels:
        if lbl == "-1":  # OOD 标签，单独处理
            continue
        mask = np.array([s == lbl for s in str_labels])
        if ood_mask is not None:
            mask = mask & ~ood_mask
        if not np.any(mask):
            continue
        fig.add_trace(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers",
            name=f"Speaker {lbl}",
            marker=dict(size=7, color=color_map[lbl],
                        line=dict(width=0.5, color="white")),
            hovertemplate=f"<b>Speaker {lbl}</b><br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra></extra>",
        ))

    # OOD 样本 (灰色 X)
    ood_total_mask = np.zeros(n, dtype=bool)
    if ood_mask is not None:
        ood_total_mask = ood_mask
    ood_label_mask = np.array([s == "-1" for s in str_labels])
    ood_total_mask = ood_total_mask | ood_label_mask

    if np.any(ood_total_mask):
        fig.add_trace(go.Scatter(
            x=coords[ood_total_mask, 0], y=coords[ood_total_mask, 1],
            mode="markers",
            name="OOD (库外)",
            marker=dict(size=9, symbol="x", color="rgba(128,128,128,0.6)",
                        line=dict(width=1.5, color="rgba(80,80,80,0.8)")),
            hovertemplate="<b>OOD样本</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="UMAP Dim 1", yaxis_title="UMAP Dim 2",
        plot_bgcolor="rgba(246,248,250,1)", paper_bgcolor="white",
        legend=dict(x=1.02, y=0.5, orientation="v"),
        height=500,
        margin=dict(l=50, r=130, t=60, b=50),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.5)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.5)")
    return fig


# ──────────────────────────────────────────────
# 相似度矩阵热力图
# ──────────────────────────────────────────────

def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    title: str = "相似度矩阵",
) -> go.Figure:
    """绘制相似度/混淆矩阵热力图"""
    n_rows, n_cols = similarity_matrix.shape
    row_labels = row_labels or [str(i) for i in range(n_rows)]
    col_labels = col_labels or [str(i) for i in range(n_cols)]

    # 显示数值注解（矩阵较小时）
    text = None
    if n_rows <= 20 and n_cols <= 20:
        text = [[f"{similarity_matrix[i,j]:.2f}" for j in range(n_cols)] for i in range(n_rows)]

    fig = go.Figure(go.Heatmap(
        z=similarity_matrix,
        x=col_labels, y=row_labels,
        text=text, texttemplate="%{text}" if text else "",
        colorscale="RdBu", zmid=0,
        colorbar=dict(title="相似度"),
    ))
    adaptive_height = min(900, max(320, 70 + n_rows * 18))

    fig.update_layout(
        title=title,
        height=adaptive_height,
        autosize=True,
        xaxis_title="支撑集 (Prototype)",
        yaxis_title="测试样本",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(automargin=True, tickangle=-35)
    fig.update_yaxes(automargin=True, tickmode="array", tickfont=dict(size=10))
    return fig


# ──────────────────────────────────────────────
# 训练损失曲线（若有记录）
# ──────────────────────────────────────────────

def plot_loss_curve(loss_curve: List[float], lr: float, epochs: int) -> go.Figure:
    """绘制训练损失曲线"""
    fig = go.Figure(go.Scatter(
        x=list(range(1, len(loss_curve) + 1)),
        y=loss_curve,
        mode="lines+markers",
        line=dict(color="rgba(66,133,244,0.9)", width=2),
        marker=dict(size=4),
        name="训练损失",
    ))
    fig.update_layout(
        title=f"训练损失曲线  (lr={lr}, epochs={epochs})",
        xaxis_title="Epoch", yaxis_title="Loss",
        plot_bgcolor="rgba(246,248,250,1)",
        height=300,
        margin=dict(l=60, r=30, t=60, b=50),
    )
    fig.add_hline(y=min(loss_curve), line_dash="dot",
                  line_color="rgba(52,168,83,0.7)",
                  annotation_text=f"最低 {min(loss_curve):.4f}")
    return fig


# ──────────────────────────────────────────────
# 综合汇总仪表板
# ──────────────────────────────────────────────

def plot_summary_gauges(metrics: Dict, mode: str = "standard") -> go.Figure:
    """绘制核心指标仪表盘 (Indicator 图)"""
    if mode in ("standard", "baseline"):
        indicators = [
            ("总错误率 (%)", metrics.get("total_err_rate", 0), 100, "lower_is_better"),
            ("平均 EER (%)", metrics.get("mean_eer", 0), 100, "lower_is_better"),
        ]
    else:
        indicators = [
            ("ID内部错误率 (%)", metrics.get("mean_id_class_err_rate", 0), 100, "lower_is_better"),
            ("ID误拒率 (%)", metrics.get("mean_id_false_reject_rate", 0), 100, "lower_is_better"),
            ("OOD误接率 (%)", metrics.get("mean_ood_err_rate", 0), 100, "lower_is_better"),
            ("平均 AUROC", metrics.get("mean_auroc", 0), 1, "higher_is_better"),
        ]

    n = len(indicators)
    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{"type": "indicator"}] * n],
    )
    for col, (label, value, max_val, direction) in enumerate(indicators, start=1):
        color = "green" if (
            (direction == "lower_is_better" and value < max_val * 0.3) or
            (direction == "higher_is_better" and value > 0.8)
        ) else ("orange" if (
            (direction == "lower_is_better" and value < max_val * 0.6) or
            (direction == "higher_is_better" and value > 0.6)
        ) else "red")

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": label, "font": {"size": 12}},
            number={"font": {"size": 22}},
            gauge={
                "axis": {"range": [0, max_val]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, max_val * 0.3], "color": "rgba(52,168,83,0.15)"},
                    {"range": [max_val * 0.3, max_val * 0.6], "color": "rgba(251,188,5,0.15)"},
                    {"range": [max_val * 0.6, max_val], "color": "rgba(234,67,53,0.15)"},
                ] if direction == "lower_is_better" else [
                    {"range": [0, max_val * 0.4], "color": "rgba(234,67,53,0.15)"},
                    {"range": [max_val * 0.4, max_val * 0.7], "color": "rgba(251,188,5,0.15)"},
                    {"range": [max_val * 0.7, max_val], "color": "rgba(52,168,83,0.15)"},
                ],
            },
        ), row=1, col=col)

    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
    return fig