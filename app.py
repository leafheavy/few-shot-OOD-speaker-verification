"""
app.py — 基于小样本学习的开放集说话人识别系统 · 前端主程序
广东工业大学 · 计算机学院 · 人工智能 22级2班 · 叶重 (3122005057)

系统功能:
  Step 1  数据集构建   (xz_data_few_shot_construction + generate_wav_list)
  Step 2  特征提取     (process_all_families via 3D-Speaker)
  Step 3  小样本学习   (baseline / few_shot_learning / OOD)
  Step 4  结果可视化   (ER/EER/AUROC、UMAP、相似度矩阵)
"""

import sys
import os
import json
import time
import threading
import numpy as np
import streamlit as st
from pathlib import Path

# ── 把 frontend 目录自身加入 path ──
_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

# ──────────────────────────────────────────────
# 页面配置
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="开放集说话人识别系统",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title{font-size:1.75rem;font-weight:700;
  background:linear-gradient(90deg,#1a73e8,#34a853);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.step-card{background:#f8f9fa;border-radius:10px;
  padding:14px 18px;border-left:4px solid #1a73e8;margin:6px 0}
.ok{color:#34a853;font-weight:600}
.warn{color:#fbbc05;font-weight:600}
.err{color:#ea4335;font-weight:600}
.metric-badge{display:inline-block;background:#e8f0fe;color:#1a73e8;
  border-radius:16px;padding:3px 12px;font-size:.85rem;font-weight:600;margin:2px}
pre.log-box{background:#1e1e1e;color:#d4d4d4;font-size:.75rem;
  border-radius:6px;padding:12px;max-height:280px;overflow-y:auto;
  white-space:pre-wrap;word-wrap:break-word}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 会话状态初始化
# ──────────────────────────────────────────────

def _init():
    defaults = {
        # 路径配置
        "user_code_dir":    str(Path.home()),
        "dataset_root":     "",
        "script_3d_speaker": "",
        "model_id":         "iic/speech_eres2netv2_sv_zh-cn_16k-common",
        "csv_files":        [],
        "meta_dir":         "",
        "audio_dir":        "",
        "output_base":      "",
        # 模块句柄
        "modules":          None,
        "modules_loaded_from": "",
        # 训练结果
        "family_results":   [],
        "summary":          {},
        "mode":             "standard",
        # UI 日志
        "step1_log":        [],
        "step2_log":        [],
        "step3_running":    False,
        "step3_progress":   0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ──────────────────────────────────────────────
# 侧边栏 — 路径与全局配置
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ 全局配置")
    st.markdown("---")

    # ── 用户代码目录 ──
    st.markdown("**📂 用户代码目录**")
    user_code_dir = st.text_input(
        "包含 few_shot_learning.py 等脚本的目录",
        value=st.session_state["user_code_dir"],
        key="_ucd",
        help="将该目录加入 sys.path，前端将从此处导入您的模型类",
    )
    st.session_state["user_code_dir"] = user_code_dir

    if st.button("🔗 加载用户代码模块", use_container_width=True):
        with st.spinner("导入模块中..."):
            from components.backend_bridge import import_user_modules
            mods = import_user_modules(user_code_dir)
            st.session_state["modules"] = mods
            st.session_state["modules_loaded_from"] = user_code_dir
        ok_mods = [k for k, v in mods.items() if v is not None]
        fail_mods = [k for k, v in mods.items() if v is None]
        if ok_mods:
            st.success(f"✅ 已加载: {', '.join(ok_mods)}")
        if fail_mods:
            st.warning(f"⚠️ 未能加载: {', '.join(fail_mods)}")

    # 显示模块状态
    mods = st.session_state["modules"]
    if mods:
        with st.expander("模块加载状态"):
            for k, v in mods.items():
                icon = "✅" if v else "❌"
                st.caption(f"{icon} {k}")

    st.markdown("---")

    # ── 数据集根目录 ──
    st.markdown("**📁 数据集根目录**")
    dataset_root = st.text_input(
        "few-shot 数据集根路径 (如 VoxCeleb/5shot3way)",
        value=st.session_state["dataset_root"],
        key="_dr",
    )
    st.session_state["dataset_root"] = dataset_root

    if dataset_root:
        root_path = Path(dataset_root)
        families = sorted(root_path.glob("family*")) if root_path.exists() else []
        n_emb = sum(1 for _ in root_path.glob("family*/embedding/train/*.npy")) if root_path.exists() else 0
        st.caption(f"Family 数: **{len(families)}** | Train embedding: **{n_emb}**")

    st.markdown("---")

    # ── 3D-Speaker 脚本路径 ──
    st.markdown("**🤖 特征提取配置**")
    script_3d = st.text_input(
        "infer_sv_batch.py 路径",
        value=st.session_state["script_3d_speaker"],
        key="_s3d",
        placeholder="/home/user/3D-Speaker/speakerlab/bin/infer_sv_batch.py",
    )
    st.session_state["script_3d_speaker"] = script_3d

    model_id = st.text_input(
        "模型 ID",
        value=st.session_state["model_id"],
        key="_mid",
    )
    st.session_state["model_id"] = model_id

    st.markdown("---")
    st.caption("广东工业大学 · 叶重 (3122005057) · 2026")


# ──────────────────────────────────────────────
# 主标题
# ──────────────────────────────────────────────

st.markdown('<h1 class="main-title">🔊 开放集说话人识别系统</h1>', unsafe_allow_html=True)

st.markdown(
    "基于 **原型网络 + 加性Softmax + 熵正则化** 的小样本说话人验证 | "
    "数据集: **VoxCeleb 系列** | 特征提取: **ERes2NetV2** (3D-Speaker)"
)

st.markdown('<hr style="margin:8px 0 16px">', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🏗️ Step 1 · 数据集构建",
    "🔬 Step 2 · 特征提取",
    "🧠 Step 3 · 小样本学习 & 评估",
    "📊 Step 4 · 结果可视化",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 · 数据集构建
# ══════════════════════════════════════════════════════════════

with tab1:
    st.subheader("🏗️ Step 1 · Few-Shot 数据集构建")
    st.markdown("""
    执行 **`few_shot_dataset_construction.py`** 将 VoxCeleb 原始音频按 family 组织，
    再执行 **`generate_wav_list.py`** 为每个 family 生成 WAV 列表文件。
    """)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### 1.1 数据集构建参数")
        meta_dir = st.text_input("meta 目录 (CSV 文件所在路径)", 
                                  value=st.session_state["meta_dir"],
                                  placeholder="/data/voxceleb/meta")
        st.session_state["meta_dir"] = meta_dir

        audio_dir = st.text_input("audio 目录 (WAV 文件所在路径)",
                                   value=st.session_state["audio_dir"],
                                   placeholder="/data/voxceleb/audio")
        st.session_state["audio_dir"] = audio_dir

        output_base = st.text_input("输出目录 (few-shot 数据集根目录)",
                                     value=st.session_state["output_base"] or dataset_root,
                                     placeholder="/data/voxceleb/5shot3way")
        st.session_state["output_base"] = output_base

        # CSV 文件选择
        if meta_dir and Path(meta_dir).exists():
            csv_candidates = sorted(Path(meta_dir).glob("*.csv"))
            csv_names = [f.name for f in csv_candidates]
            selected_csvs = st.multiselect(
                "选择要使用的 CSV 文件",
                options=csv_names,
                default=csv_names[:3] if len(csv_names) >= 3 else csv_names,
            )
        else:
            st.info("请填写 meta 目录路径后自动列出 CSV 文件")
            selected_csvs = []

        col_a, col_b = st.columns(2)
        speakers_per_family = col_a.number_input("每 family 说话人数", 3, 20, 5)
        shot_num = col_b.number_input("每人支撑样本数 (shot)", 1, 10, 3)

        st.markdown("---")
        run_build = st.button("▶️ 执行数据集构建", type="primary", use_container_width=True)

    with c2:
        st.markdown("#### 构建日志")
        log_box = st.empty()

        def refresh_log1():
            log_text = "\n".join(st.session_state["step1_log"][-80:])
            log_box.markdown(f'<pre class="log-box">{log_text}</pre>', unsafe_allow_html=True)

        refresh_log1()

        if run_build:
            if not (meta_dir and audio_dir and output_base):
                st.error("请填写 meta目录、audio目录、输出目录")
            else:
                st.session_state["step1_log"] = ["[系统] 开始执行数据集构建..."]
                refresh_log1()

                # ── 调用构建脚本（通过命令行参数注入配置） ──
                try:
                    script_path = Path(user_code_dir) / "few_shot_dataset_construction.py"
                    if not script_path.exists():
                        st.error(f"未找到 few_shot_dataset_construction.py 于 {user_code_dir}")
                    else:
                        src = script_path.read_text()
                        # 替换配置变量
                        src = src.replace(
                            'meta_dir = "/data/voxceleb/meta"', f'meta_dir = "{meta_dir}"'
                        ).replace(
                            'audio_dir = "/data/voxceleb/audio"', f'audio_dir = "{audio_dir}"'
                        ).replace(
                            'output_base = "/data/voxceleb/5shot3way"', f'output_base = "{output_base}"'
                        )
                        # 替换 CSV 列表
                        csv_list_str = str(selected_csvs)
                        src = src.replace(
                            'csv_files = [\n    "messages_data_20250710_195241_part_001.csv"',
                            f'csv_files = {csv_list_str}\n# patched'
                        )
                        src = src.replace(
                            f'speakers_per_family = 5', f'speakers_per_family = {speakers_per_family}'
                        ).replace(
                            f'shot_num = 3', f'shot_num = {shot_num}'
                        )

                        from utils.subprocess_runner import LiveProcessRunner

                        cmd = [
                            sys.executable,
                            str(script_path),
                            "--meta-dir", meta_dir,
                            "--audio-dir", audio_dir,
                            "--output-base", output_base,
                            "--speakers-per-family", str(speakers_per_family),
                            "--shot-num", str(shot_num),
                            "--dataset-name", "voxceleb",
                        ]
                        if selected_csvs:
                            cmd.extend(["--csv-files", *selected_csvs])

                        runner = LiveProcessRunner(cmd, cwd=user_code_dir)

                        runner.start()
                        prog = st.progress(0, "构建数据集...")
                        for i, line in enumerate(runner.iter_output()):
                            st.session_state["step1_log"].append(line)
                            prog.progress(min((i + 1) * 0.02, 0.95))
                            refresh_log1()
                        rc = runner.wait()
                        prog.progress(1.0)

                        if rc == 0:
                            st.session_state["step1_log"].append("[系统] ✅ 数据集构建完成!")
                            st.success("✅ 数据集构建成功")
                        else:
                            st.session_state["step1_log"].append(f"[系统] ❌ 退出码: {rc}")
                            st.error(f"脚本异常退出 (code={rc})")
                        refresh_log1()

                except Exception as ex:
                    st.error(f"执行出错: {ex}")
                    st.session_state["step1_log"].append(f"[错误] {ex}")
                    refresh_log1()

    # ── 1.2 生成 WAV 列表 ──
    st.markdown("---")
    st.markdown("#### 1.2 生成 WAV 列表文件")
    col_w1, col_w2 = st.columns([1, 1])

    with col_w1:
        wav_list_root = st.text_input(
            "数据集根目录 (generate_wav_list 输入)",
            value=output_base or dataset_root,
            key="wl_root",
        )
        run_wav = st.button("▶️ 生成 WAV 列表", use_container_width=True)

    with col_w2:
        wav_log_box = st.empty()

        if run_wav:
            if not wav_list_root:
                st.error("请输入数据集根目录")
            else:
                script_wav = Path(user_code_dir) / "generate_wav_list.py"
                if not script_wav.exists():
                    st.error(f"未找到 generate_wav_list.py 于 {user_code_dir}")
                else:
                    from utils.subprocess_runner import LiveProcessRunner
                    runner = LiveProcessRunner(
                        [sys.executable, str(script_wav),
                         "--dataset_root", wav_list_root]
                    )
                    runner.start()
                    lines = []
                    for line in runner.iter_output():
                        lines.append(line)
                        wav_log_box.markdown(
                            f'<pre class="log-box">{"chr(10)".join(lines[-30:])}</pre>',
                            unsafe_allow_html=True
                        )
                    rc = runner.wait()
                    if rc == 0:
                        st.success("✅ WAV 列表生成完成")
                    else:
                        st.error(f"生成失败 (code={rc})")

    # ── 数据集状态检查 ──
    st.markdown("---")
    st.markdown("#### 📋 数据集状态")
    check_root = output_base or dataset_root
    if check_root and Path(check_root).exists():
        root_p = Path(check_root)
        families = sorted(root_p.glob("family*"))
        n_fam = len(families)
        n_train_wav = sum(1 for _ in root_p.glob("family*/train/*/*.wav"))
        n_test_wav = sum(1 for _ in root_p.glob("family*/test/*/*.wav"))
        n_train_emb = sum(1 for _ in root_p.glob("family*/embedding/train/*.npy"))
        n_test_emb = sum(1 for _ in root_p.glob("family*/embedding/test/*.npy"))
        n_wav_list = sum(1 for _ in root_p.glob("family*/train_wav_list.txt"))

        cols = st.columns(5)
        cols[0].metric("Family 数", n_fam)
        cols[1].metric("Train WAV", n_train_wav)
        cols[2].metric("Test WAV", n_test_wav)
        cols[3].metric("Train Embedding", n_train_emb)
        cols[4].metric("WAV 列表文件", n_wav_list)

        if n_fam > 0:
            # 抽样展示前5个family
            st.markdown("**前 5 个 Family 状态:**")
            for fam in families[:5]:
                n_tr = len(list(fam.glob("embedding/train/*.npy")))
                n_te = len(list(fam.glob("embedding/test/*.npy")))
                has_list = (fam / "train_wav_list.txt").exists()
                icon_emb = "✅" if n_tr > 0 else "❌"
                icon_list = "✅" if has_list else "⚠️"
                st.caption(
                    f"{icon_emb} `{fam.name}` — "
                    f"train_emb: **{n_tr}** | test_emb: **{n_te}** | "
                    f"wav_list: {icon_list}"
                )
    else:
        st.info("请在侧边栏或上方填写数据集根目录")


# ══════════════════════════════════════════════════════════════
# TAB 2 · 特征提取
# ══════════════════════════════════════════════════════════════

with tab2:
    st.subheader("🔬 Step 2 · 特征提取 (ERes2NetV2 via 3D-Speaker)")
    st.markdown("""
    调用 `process_all_families.py` 批量对所有 family 的 WAV 文件提取 `.npy` 嵌入向量。  
    **前提:** 已完成 Step 1，且 3D-Speaker 环境与 `infer_sv_batch.py` 路径已配置。
    """)

    c_ext1, c_ext2 = st.columns([1, 1.2])

    with c_ext1:
        st.markdown("#### 提取参数")
        ext_dataset = st.text_input(
            "数据集根目录", value=dataset_root or st.session_state["output_base"], key="ext_ds"
        )
        ext_script = st.text_input(
            "infer_sv_batch.py 路径", value=script_3d or st.session_state["script_3d_speaker"],
            key="ext_sc"
        )
        ext_model = st.text_input(
            "模型 ID", value=model_id, key="ext_mid"
        )
        num_workers = st.slider("并行 Worker 数", 1, 8, 1)

        # 前提检查
        checks = {
            "数据集目录存在": Path(ext_dataset).exists() if ext_dataset else False,
            "3D-Speaker 脚本存在": Path(ext_script).exists() if ext_script else False,
            "用户代码已加载": st.session_state["modules"] is not None,
        }
        for chk, ok in checks.items():
            icon = "✅" if ok else "❌"
            st.caption(f"{icon} {chk}")

        run_ext = st.button("▶️ 开始特征提取", type="primary",
                             use_container_width=True,
                             disabled=not all(checks.values()))

    with c_ext2:
        st.markdown("#### 提取日志")
        ext_log_box = st.empty()
        ext_prog = st.empty()

        if run_ext:
            script_paf = Path(user_code_dir) / "process_all_families.py"
            if not script_paf.exists():
                st.error(f"未找到 process_all_families.py 于 {user_code_dir}")
            else:
                from utils.subprocess_runner import LiveProcessRunner
                cmd = [
                    sys.executable, str(script_paf),
                    "--dataset_root", ext_dataset,
                    "--model_id", ext_model,
                    "--script_path", ext_script,
                ]
                runner = LiveProcessRunner(cmd)
                runner.start()
                log_lines = []
                pbar = ext_prog.progress(0, "提取特征中...")
                for i, line in enumerate(runner.iter_output()):
                    log_lines.append(line)
                    ext_log_box.markdown(
                        f'<pre class="log-box">{"chr(10)".join(log_lines[-40:])}</pre>',
                        unsafe_allow_html=True
                    )
                    pbar.progress(min((i + 1) * 0.005, 0.95))
                rc = runner.wait()
                pbar.progress(1.0)
                if rc == 0:
                    st.success("✅ 特征提取完成!")
                else:
                    st.error(f"提取异常退出 (code={rc})")

    # ── 提取进度统计 ──
    st.markdown("---")
    if ext_dataset and Path(ext_dataset).exists():
        st.markdown("#### 📊 嵌入提取进度")
        root_p = Path(ext_dataset)
        families = sorted(root_p.glob("family*"))
        total = len(families)
        done = sum(1 for f in families
                   if any(f.glob("embedding/train/*.npy")) and any(f.glob("embedding/test/*.npy")))
        partial = sum(1 for f in families
                      if any(f.glob("embedding/train/*.npy")) or any(f.glob("embedding/test/*.npy"))) - done
        pending = total - done - partial

        p1, p2, p3 = st.columns(3)
        p1.metric("✅ 完成", done, delta=f"{done/max(total,1)*100:.0f}%")
        p2.metric("⚠️ 部分完成", partial)
        p3.metric("⏳ 待提取", pending)

        if total > 0:
            st.progress(done / total, text=f"{done}/{total} families 完成特征提取")


# ══════════════════════════════════════════════════════════════
# TAB 3 · 小样本学习 & 评估
# ══════════════════════════════════════════════════════════════

with tab3:
    st.subheader("🧠 Step 3 · 小样本学习与评估")

    # ── 参数配置面板 ──
    cfg_col, run_col = st.columns([1, 1.5])

    with cfg_col:
        st.markdown("#### 🔧 训练参数")

        mode = st.radio(
            "运行模式",
            options=["baseline", "standard", "ood"],
            format_func=lambda x: {
                "baseline": "📏 Baseline (余弦相似度，无学习)",
                "standard": "🧠 Few-Shot Learning (原型网络)",
                "ood": "🔍 OOD-Aware (原型网络 + 开放集检测)",
            }[x],
            index=["baseline", "standard", "ood"].index(
                st.session_state.get("mode", "standard")
            ),
        )
        st.session_state["mode"] = mode

        if mode != "baseline":
            st.markdown("---")
            epochs = st.slider("训练轮数 (Epochs)", 10, 300, 100, 10)
            lr = st.select_slider(
                "学习率 (lr)",
                options=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
                value=1e-3,
                format_func=lambda x: f"{x:.0e}",
            )
        else:
            epochs, lr = 0, 0.0

        if mode == "ood":
            ood_threshold = st.slider(
                "OOD 置信度阈值",
                min_value=0.1, max_value=0.9, value=0.4, step=0.05,
                help="低于此阈值的样本视为库外 (OOD)",
            )
        else:
            ood_threshold = 0.4

        st.markdown("---")
        batch_size = st.select_slider("Batch Size", options=[4, 8, 16, 32], value=8)
        preload = st.checkbox("预加载 embedding 到内存", value=True)

        # 数据集范围
        eval_dataset = st.text_input(
            "评估数据集路径",
            value=dataset_root or st.session_state["output_base"],
            key="eval_ds",
        )

        max_families = st.number_input(
            "最多评估 Family 数 (0=全部)", 0, 9999, 0,
            help="设为0则评估全部family，设置较小值可快速预览结果",
        )

        # 前提检查
        mods = st.session_state["modules"]
        checks3 = {
            "数据集路径存在": Path(eval_dataset).exists() if eval_dataset else False,
            "模块已加载": mods is not None,
        }
        if mods:
            key = {"baseline": "baseline", "standard": "few_shot_learning",
                   "ood": "few_shot_learning_OOD_other_loss_xz"}.get(mode, "few_shot_learning")
            checks3[f"'{key}' 模块可用"] = mods.get(key) is not None

        for chk, ok in checks3.items():
            icon = "✅" if ok else "❌"
            st.caption(f"{icon} {chk}")

        run_eval = st.button(
            "▶️ 开始评估", type="primary", use_container_width=True,
            disabled=not all(checks3.values()),
        )

    with run_col:
        st.markdown("#### 📈 实时进度")
        prog_bar = st.empty()
        prog_text = st.empty()
        result_placeholder = st.empty()

        if run_eval:
            from components.backend_bridge import (
                FewShotRunner, create_family_loaders, import_user_modules
            )
            from utils.metrics import aggregate_family_results

            mods = st.session_state["modules"]
            if mods is None:
                st.error("请先在侧边栏加载用户代码模块")
                st.stop()

            root_p = Path(eval_dataset)
            families = sorted(root_p.glob("family*"))
            if max_families > 0:
                families = families[:max_families]

            if not families:
                st.error("未找到任何 family 目录，请检查数据集路径")
                st.stop()

            # 加载数据
            with st.spinner("加载 DataLoader..."):
                try:
                    support_loaders, test_loaders = create_family_loaders(
                        eval_dataset, mods, mode=mode,
                        batch_size=batch_size, preload=preload
                    )
                except Exception as e:
                    st.error(f"DataLoader 创建失败: {e}")
                    st.stop()

            runner = FewShotRunner(mods, mode=mode)
            if not runner.is_available():
                st.error("模型类未能初始化，请检查模块是否正确加载")
                st.stop()

            family_results = []
            total = len([f for f in families if f.name in support_loaders])
            processed = 0

            for family in families:
                fname = family.name
                if fname not in support_loaders or fname not in test_loaders:
                    continue

                prog_bar.progress(processed / max(total, 1),
                                  text=f"处理 {fname} ({processed+1}/{total})")

                try:
                    res = runner.run_family(
                        support_loaders[fname], test_loaders[fname],
                        epochs=epochs, lr=lr, ood_threshold=ood_threshold,
                    )
                    res["family_name"] = fname
                    family_results.append(res)
                except Exception as e:
                    family_results.append({
                        "family_name": fname, "error": str(e),
                        "err_rate": 0.0, "eer": 0.0,
                    })
                    st.warning(f"⚠️ {fname}: {e}")

                processed += 1

                # 实时汇总显示
                tmp_summary = aggregate_family_results(family_results)
                with result_placeholder.container():
                    m1, m2 = st.columns(2)
                    m1.metric("已处理 Family", processed)
                    m2.metric("当前平均 EER", f"{tmp_summary.get('mean_eer', 0):.2f}%")
                    if mode == "ood":
                        m3, m4 = st.columns(2)
                        m3.metric("AUROC", f"{tmp_summary.get('mean_auroc', 0):.4f}")
                        m4.metric("总错误率", f"{tmp_summary.get('total_err_rate', 0):.2f}%")

            prog_bar.progress(1.0, text="✅ 评估完成!")

            # 保存结果
            st.session_state["family_results"] = family_results
            summary = aggregate_family_results(family_results)
            st.session_state["summary"] = summary

            st.success(f"✅ 评估完成！共处理 {len(family_results)} 个 families")

    # ── 当前结果摘要 ──
    if st.session_state["family_results"]:
        st.markdown("---")
        st.markdown("#### 📋 评估结果摘要")
        s = st.session_state["summary"]
        cur_mode = st.session_state["mode"]

        if cur_mode == "ood":
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("总错误率", f"{s.get('total_err_rate',0):.2f}%")
            c2.metric("平均 EER", f"{s.get('mean_eer',0):.2f}%")
            c3.metric("ID内部错误率", f"{s.get('mean_id_class_err_rate',0):.2f}%")
            c4.metric("ID误拒率", f"{s.get('mean_id_false_reject_rate',0):.2f}%")
            c5.metric("AUROC", f"{s.get('mean_auroc',0):.4f}")
            c6.metric("FPR@95", f"{s.get('mean_fpr95',0):.4f}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("总错误率", f"{s.get('total_err_rate',0):.2f}%")
            c2.metric("平均 EER", f"{s.get('mean_eer',0):.2f}%")
            c3.metric("评估 Family 数", s.get("n_families", 0))

        # 详细 Family 表格
        with st.expander("📊 各 Family 详细结果", expanded=False):
            import pandas as pd
            rows = []
            for r in st.session_state["family_results"]:
                row = {
                    "Family": r.get("family_name", ""),
                    "错误率 (%)": f"{r.get('err_rate', 0):.2f}",
                    "EER (%)": f"{r.get('eer', 0)*100:.2f}",
                    "EER阈值": f"{r.get('eer_threshold', 0):.4f}",
                }
                if "auroc" in r:
                    row.update({
                        "AUROC": f"{r.get('auroc', 0):.4f}",
                        "FPR@95": f"{r.get('fpr_at_95', 0):.4f}",
                        "OOD误接率(%)": f"{r.get('ood_err_rate', 0):.2f}",
                        "ID误拒率(%)": f"{r.get('id_false_reject_rate', 0):.2f}",
                    })
                if "error" in r:
                    row["备注"] = f"❌ {r['error'][:40]}"
                rows.append(row)
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # 导出按钮
            csv_str = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ 导出 CSV",
                data=csv_str,
                file_name=f"sv_results_{cur_mode}_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════
# TAB 4 · 结果可视化
# ══════════════════════════════════════════════════════════════

with tab4:
    st.subheader("📊 Step 4 · 结果可视化")

    family_results = st.session_state.get("family_results", [])
    summary = st.session_state.get("summary", {})
    cur_mode = st.session_state.get("mode", "standard")

    if not family_results:
        st.info("请先在 Step 3 完成模型评估，结果将自动在此处呈现。")
    else:
        from components.visualization import (
            plot_family_metrics, plot_ood_radar,
            plot_summary_gauges, plot_similarity_heatmap,
            plot_umap, plot_loss_curve,
        )

        # ── 仪表盘 ──
        st.markdown("#### 🎯 核心指标仪表盘")
        fig_gauge = plot_summary_gauges(summary, mode=cur_mode)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown('<hr style="margin:8px 0">', unsafe_allow_html=True)

        # ── Family 级别柱状图 ──
        st.markdown("#### 📊 各 Family 评估指标")
        fig_fam = plot_family_metrics(family_results, mode=cur_mode)
        st.plotly_chart(fig_fam, use_container_width=True)

        if cur_mode == "ood":
            st.markdown("#### 🕸️ OOD 综合能力雷达图")
            fig_radar = plot_ood_radar(summary)
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown('<hr style="margin:8px 0">', unsafe_allow_html=True)

        # ── UMAP 可视化 ──
        st.markdown("#### 🗺️ UMAP 嵌入空间可视化")

        vis_col1, vis_col2 = st.columns([1, 3])
        with vis_col1:
            # Family 选择
            valid_families = [r["family_name"] for r in family_results
                              if "similarity_matrix" in r]
            if valid_families:
                sel_family = st.selectbox("选择 Family", valid_families)
                umap_n_neighbors = st.slider("UMAP 邻居数", 2, 20, 8)
                umap_min_dist = st.slider("UMAP 最小距离", 0.05, 0.8, 0.3, 0.05)
                show_umap = st.button("🔄 绘制 UMAP", use_container_width=True)
            else:
                st.info("无可视化数据")
                sel_family = None
                show_umap = False

        with vis_col2:
            if show_umap and sel_family:
                sel_result = next(
                    (r for r in family_results if r["family_name"] == sel_family), None
                )
                if sel_result and "similarity_matrix" in sel_result:
                    sim_mat = sel_result["similarity_matrix"]
                    true_labels = sel_result["true_labels"]

                    # 用相似度矩阵的行作为嵌入近似展示
                    embs = sim_mat.numpy()
                    labels = true_labels.numpy()
                    ood_mask = labels < 0

                    with st.spinner("UMAP 降维计算中..."):
                        try:
                            fig_umap = plot_umap(
                                embs, labels,
                                title=f"UMAP — {sel_family} (相似度空间)",
                                n_neighbors=umap_n_neighbors,
                                min_dist=umap_min_dist,
                                ood_mask=ood_mask if np.any(ood_mask) else None,
                            )
                            if fig_umap:
                                st.plotly_chart(fig_umap, use_container_width=True)
                            else:
                                st.warning("样本数不足，无法绘制 UMAP")
                        except Exception as e:
                            st.error(f"UMAP 绘制失败: {e}")

        st.markdown('<hr style="margin:8px 0">', unsafe_allow_html=True)

        # ── 相似度矩阵热力图 ──
        st.markdown("#### 🔥 相似度矩阵热力图")
        heat_col1, heat_col2 = st.columns([1, 3])

        with heat_col1:
            valid_for_heat = [r["family_name"] for r in family_results
                              if "similarity_matrix" in r]
            if valid_for_heat:
                heat_family = st.selectbox("选择 Family", valid_for_heat, key="heat_sel")
                show_heat = st.button("🔄 绘制热力图", use_container_width=True)
            else:
                heat_family = None
                show_heat = False

        with heat_col2:
            if show_heat and heat_family:
                hr = next((r for r in family_results if r["family_name"] == heat_family), None)
                if hr and "similarity_matrix" in hr:
                    sim_mat = hr["similarity_matrix"].numpy()
                    true_labels_np = hr["true_labels"].numpy()
                    # 按真实标签排序
                    sort_idx = np.argsort(true_labels_np)
                    sim_sorted = sim_mat[sort_idx, :]
                    row_labels = [
                        f"{'OOD' if true_labels_np[i] < 0 else f'spk{true_labels_np[i]}'}"
                        for i in sort_idx
                    ]
                    col_labels = [f"proto{j}" for j in range(sim_mat.shape[1])]
                    fig_heat = plot_similarity_heatmap(
                        sim_sorted, row_labels, col_labels,
                        title=f"相似度矩阵 — {heat_family}",
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

    # ── 历史结果文件加载 ──
    st.markdown("---")
    st.markdown("#### 📂 加载历史结果文件 (.txt)")
    st.markdown("支持加载 `few_shot_learning_result*.txt` 或 `baseline_result*.txt` 格式")

    uploaded_result = st.file_uploader(
        "上传结果文件", type=["txt"], key="result_upload"
    )
    if uploaded_result:
        content = uploaded_result.read().decode("utf-8")
        lines = content.strip().split("\n")
        st.code(content[-3000:], language="text")

        # 解析关键指标
        import re
        err_pattern = re.compile(r"错误率为: ([\d.]+) %")
        eer_pattern = re.compile(r"EER为: ([\d.]+) %")
        errors = err_pattern.findall(content)
        eers_parsed = eer_pattern.findall(content)

        if errors:
            import pandas as pd
            n = min(len(errors), len(eers_parsed))
            df_hist = pd.DataFrame({
                "Family": [f"family{i+1:05d}" for i in range(n)],
                "错误率(%)": [float(e) for e in errors[:n]],
                "EER(%)": [float(e) for e in eers_parsed[:n]],
            })
            st.dataframe(df_hist, use_container_width=True, hide_index=True)

            from components.visualization import plot_family_metrics
            fig_hist = plot_family_metrics(
                [{"family_name": r["Family"],
                  "err_rate": r["错误率(%)"],
                  "eer": r["EER(%)"] / 100}
                 for _, r in df_hist.iterrows()]
            )
            st.plotly_chart(fig_hist, use_container_width=True)


# ──────────────────────────────────────────────
# 页脚
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#999;font-size:.78rem'>"
    "广东工业大学 · 计算机学院 · 人工智能专业 22级2班 · "
    "基于小样本学习的开放集说话人识别系统 · 叶重 (3122005057) · 指导教师: 胡宇 · 2026"
    "</div>",
    unsafe_allow_html=True,
)
