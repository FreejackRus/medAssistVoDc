# gpu_monitor.py
from __future__ import annotations
import threading
import time
from typing import Callable

try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        NVMLError
    )

    nvmlInit()
    _handle = nvmlDeviceGetHandleByIndex(0)

    def vram_now() -> tuple[int, int]:
        """(used_MB, total_MB)"""
        try:
            info = nvmlDeviceGetMemoryInfo(_handle)
            return info.used // 1024**2, info.total // 1024**2
        except NVMLError:
            return 0, 0

    def print_vram(label: str = "VRAM"):
        used, total = vram_now()
        print(f"{label}: {used:4d} / {total:4d} MB")

    def background_monitor(interval: float = 1.0, stop_event: threading.Event | None = None):
        """Потоковый монитор."""
        while not (stop_event and stop_event.is_set()):
            print_vram("GPU")
            time.sleep(interval)

except ImportError:
    def vram_now():
        return 0, 0
    def print_vram(_=""):
        pass
    def background_monitor(*_, **__):
        pass