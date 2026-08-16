from __future__ import annotations

import resource
import time
from dataclasses import dataclass


def _max_rss_mb() -> float:
    # Linux ru_maxrss is KiB. This project targets Ubuntu/NVIDIA GPU hosts.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@dataclass
class RuntimeMeter:
    device: str

    def __post_init__(self) -> None:
        self.started_at = 0.0
        self.start_rss_mb = 0.0

    def start(self) -> None:
        try:
            import torch
        except ImportError:
            torch = None

        self.start_rss_mb = _max_rss_mb()
        if torch is not None and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.started_at = time.perf_counter()

    def stop(self, audio_seconds: float) -> dict[str, float | None]:
        try:
            import torch
        except ImportError:
            torch = None

        if torch is not None and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started_at
        result: dict[str, float | None] = {
            "audio_seconds": round(float(audio_seconds), 4),
            "wall_seconds": round(elapsed, 4),
            "rtf": round(elapsed / audio_seconds, 6) if audio_seconds else None,
            "throughput_x_realtime": round(audio_seconds / elapsed, 3) if elapsed else None,
            "peak_rss_mb": round(_max_rss_mb(), 2),
            "rss_before_inference_mb": round(self.start_rss_mb, 2),
            "peak_gpu_allocated_mb": None,
            "peak_gpu_reserved_mb": None,
        }
        if torch is not None and self.device.startswith("cuda") and torch.cuda.is_available():
            result["peak_gpu_allocated_mb"] = round(
                torch.cuda.max_memory_allocated() / (1024**2), 2
            )
            result["peak_gpu_reserved_mb"] = round(
                torch.cuda.max_memory_reserved() / (1024**2), 2
            )
        return result
