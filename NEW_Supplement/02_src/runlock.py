"""防止同一个脚本被重复启动，以及记录执行耗时。

之前出过一次事故：同一条命令被启动了两遍，两个进程抢 CPU、往同一个日志里写，
跑了 9 个多小时也没结束，机器还发烫。加一把锁来杜绝这种情况。
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

LOCK_DIR = Path("/tmp/claude_runlocks")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


@contextmanager
def single_instance(name: str):
    """同名任务只允许一个在跑。已有存活进程时直接退出，不做任何事。"""
    LOCK_DIR.mkdir(exist_ok=True)
    f = LOCK_DIR / f"{name}.pid"
    if f.exists():
        try:
            old = int(f.read_text().strip())
        except Exception:
            old = -1
        if old > 0 and _alive(old):
            raise SystemExit(
                f"[已跳过] 任务 {name} 已有进程在跑（PID {old}）。"
                f"如果确认那个进程已经死了，删掉 {f} 再重试。")
        f.unlink(missing_ok=True)
    f.write_text(str(os.getpid()))
    try:
        yield
    finally:
        f.unlink(missing_ok=True)


class Timer:
    """记录各阶段起止时间，最后能直接输出成一段可读的耗时说明。"""

    def __init__(self, name: str):
        self.name = name
        self.t0 = time.time()
        self.start = datetime.now()
        self.marks: list[tuple[str, float]] = []

    def mark(self, label: str) -> None:
        self.marks.append((label, time.time()))

    @staticmethod
    def _fmt(sec: float) -> str:
        h, m = divmod(int(sec) // 60, 60)
        return f"{h} 小时 {m} 分" if h else f"{m} 分 {int(sec) % 60} 秒"

    def report(self) -> str:
        end = datetime.now()
        total = time.time() - self.t0
        L = [f"- 任务：{self.name}",
             f"- 开始 {self.start:%Y-%m-%d %H:%M:%S}，"
             f"结束 {end:%Y-%m-%d %H:%M:%S}，共用时 **{self._fmt(total)}**"]
        prev = self.t0
        for label, t in self.marks:
            L.append(f"    - {label}：{self._fmt(t - prev)}")
            prev = t
        return "\n".join(L)
