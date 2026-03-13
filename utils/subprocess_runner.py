"""
subprocess_runner.py — 带实时输出的子进程管理器
用于在 Streamlit 界面中运行管道脚本并实时展示 stdout/stderr
"""
import subprocess
import sys
import threading
import queue
from pathlib import Path
from typing import Optional, List, Callable

IGNORED_PATTERNS = (
    "missing ScriptRunContext! This warning can be ignored when running in bare mode.",
)

class LiveProcessRunner:
    """
    在后台运行子进程并通过队列传递输出行
    
    Usage:
        runner = LiveProcessRunner(["python", "script.py", "--arg", "val"])
        runner.start()
        for line in runner.iter_output():
            st.write(line)
        exit_code = runner.wait()
    """

    def __init__(self, cmd: List[str], cwd: Optional[str] = None, env=None):
        self.cmd = cmd
        self.cwd = cwd
        self.env = env
        self._proc: Optional[subprocess.Popen] = None
        self._queue: queue.Queue = queue.Queue()
        self._done = threading.Event()

    def start(self):
        """启动子进程和输出收集线程"""
        self._proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并 stderr → stdout
            text=True,
            bufsize=1,
            cwd=self.cwd,
            env=self.env,
        )
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()

    def _reader(self):
        """后台线程：读取子进程输出行并入队"""
        try:
            for line in self._proc.stdout:
                clean_line = line.rstrip("\n")
                if any(pattern in clean_line for pattern in IGNORED_PATTERNS):
                    continue
                self._queue.put(clean_line)
        finally:
            self._proc.stdout.close()
            self._done.set()

    def iter_output(self, timeout: float = 0.1):
        """
        生成器：持续产出输出行，直到子进程结束
        
        Args:
            timeout: 每次等待的超时时间(秒)
        
        Yields:
            str: 每一行输出
        """
        while not (self._done.is_set() and self._queue.empty()):
            try:
                line = self._queue.get(timeout=timeout)
                yield line
            except queue.Empty:
                continue

    def wait(self) -> int:
        """等待子进程结束并返回退出码"""
        if self._proc:
            self._proc.wait()
            return self._proc.returncode
        return -1

    def kill(self):
        """强制终止子进程"""
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


def run_script(
    script_path: str,
    args: List[str],
    cwd: Optional[str] = None,
    output_callback: Optional[Callable[[str], None]] = None,
    python_exec: str = sys.executable,
) -> int:
    """
    同步运行脚本并收集输出
    
    Args:
        script_path:     脚本路径
        args:            命令行参数列表
        cwd:             工作目录
        output_callback: 每行输出的回调函数 (None 则打印到 stdout)
        python_exec:     Python 解释器路径
    
    Returns:
        exit_code: 退出码，0=成功
    """
    cmd = [python_exec, str(script_path)] + args
    runner = LiveProcessRunner(cmd, cwd=cwd)
    runner.start()

    for line in runner.iter_output():
        if output_callback:
            output_callback(line)
        else:
            print(line)

    return runner.wait()