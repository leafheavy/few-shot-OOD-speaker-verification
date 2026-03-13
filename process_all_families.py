import os
import warnings
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm

def _count_non_empty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def _count_npy_files(path: Path) -> int:
    return sum(1 for _ in path.rglob("*.npy"))

def _resolve_infer_script_path(script_path: str) -> Path:
    """Resolve infer_sv_batch.py path robustly for Linux/Windows style inputs."""
    raw = str(script_path).strip().strip('"').strip("'")
    normalized = raw.replace('\\', '/')

    p = Path(normalized).expanduser()
    repo_root = Path(__file__).resolve().parent

    candidates = []
    if p.is_absolute():
        candidates.append(p)
        # Handle inputs like "\\speakerlab\\bin\\" -> "/speakerlab/bin/".
        candidates.append(repo_root / normalized.lstrip('/'))
    else:
        candidates.append((Path.cwd() / p).resolve())
        candidates.append((repo_root / p).resolve())

    for cand in candidates:
        if cand.is_dir():
            cand = cand / 'infer_sv_batch.py'
        if cand.is_file():
            return cand.resolve()

    raise FileNotFoundError(
        f"未找到 infer_sv_batch.py。收到 script_path={script_path!r}。"
        "请传入文件路径（例如 /workspace/.../speakerlab/bin/infer_sv_batch.py）"
    )

def process_family_wav_lists(dataset_root, model_id, script_path):
    """
    自动处理所有Family文件夹中的train和test WAV列表
    
    Args:
        dataset_root: 数据集根路径
        model_id: 使用的模型ID
        script_path: infer_sv_batch.py脚本的路径
    """
    # 忽略Python警告
    warnings.filterwarnings("ignore")

    dataset_root = Path(dataset_root).resolve()
    families = [f for f in dataset_root.iterdir() if f.is_dir() and f.name.startswith("family")]
    
    print(f"找到 {len(families)} 个Family文件夹")
    
    # 获取脚本的绝对路径
    script_path = _resolve_infer_script_path(script_path)
    print(f"使用特征提取脚本: {script_path}", flush=True)
    
    for family in tqdm(families, desc="处理Family"):
        # 检查是否存在WAV列表文件
        train_list = (family / "train_wav_list.txt").resolve()
        test_list = (family / "test_wav_list.txt").resolve()
        
        if not train_list.exists() or not test_list.exists():
            print(f"警告: {family.name} 缺少WAV列表文件，跳过...")
            continue
        
        # 创建输出目录
        train_output_dir = family / "embedding" / "train"
        test_output_dir = family / "embedding" / "test"
        train_output_dir.mkdir(parents=True, exist_ok=True)
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        train_expected = _count_non_empty_lines(train_list)
        test_expected = _count_non_empty_lines(test_list)

        print(f"[{family.name}] 开始提取: train={train_expected}, test={test_expected}", flush=True)

        # 处理train列表
        # print(f"处理 {family.name} 的train列表...")
        cmd_train = [
            "python", str(script_path),
            "--model_id", model_id,
            "--wavs", str(train_list),
            "--feat_out_dir", str(train_output_dir),
            "--feat_out_format", "npy",
            "--diable_progress_bar"
        ]
        
        try:
            # 设置工作目录为数据集根目录，确保相对路径正确解析
            subprocess.run(cmd_train, check=True, cwd=str(dataset_root))
            # print(f"成功处理 {family.name} 的train列表")
            train_actual = _count_npy_files(train_output_dir)
            if train_actual < train_expected:
                raise RuntimeError(
                    f"{family.name} train embedding数量不足: expected={train_expected}, actual={train_actual}"
                )
            print(f"[{family.name}] train完成: {train_actual}/{train_expected}", flush=True)
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"处理 {family.name} 的train列表时出错: {e}")
            continue
        
        # 处理test列表
        # print(f"处理 {family.name} 的test列表...")
        cmd_test = [
            "python", str(script_path),
            "--model_id", model_id,
            "--wavs", str(test_list),
            "--feat_out_dir", str(test_output_dir),
            "--feat_out_format", "npy",
            "--diable_progress_bar"
        ]
        
        try:
            # 设置工作目录为数据集根目录，确保相对路径正确解析
            subprocess.run(cmd_test, check=True, cwd=str(dataset_root))
            # print(f"成功处理 {family.name} 的test列表")
            test_actual = _count_npy_files(test_output_dir)
            if test_actual < test_expected:
                raise RuntimeError(
                    f"{family.name} test embedding数量不足: expected={test_expected}, actual={test_actual}"
                )
            print(f"[{family.name}] test完成: {test_actual}/{test_expected}", flush=True)
            print(f"[{family.name}] 全部完成", flush=True)
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"处理 {family.name} 的test列表时出错: {e}")
            continue
    
    print("所有Family处理完成!")

def main():
    parser = argparse.ArgumentParser(description='自动处理所有Family的WAV列表进行特征提取')
    parser.add_argument('--dataset_root', required=True, help='数据集根路径')
    parser.add_argument('--model_id', default='iic/speech_eres2netv2_sv_zh-cn_16k-common', 
                       help='使用的模型ID')
    parser.add_argument('--script_path', default='/home/ps/3D-Speaker/speakerlab/bin/infer_sv_batch.py', 
                       help='infer_sv_batch.py脚本的路径')
    
    args = parser.parse_args()
    
    # 执行处理
    process_family_wav_lists(
        dataset_root=args.dataset_root,
        model_id=args.model_id,
        script_path=args.script_path
    )

if __name__ == "__main__":
    main()