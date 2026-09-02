"""
hip_quant.smi — easy GPU usage query while ML programs are running

Uses vendored `gpu-smi` Rust binary when available (913KB single-exe,
HIP→ADL→WMI→sysfs, no ROCm SDK needed). Falls back to torch/HIP if binary
missing so `pip install hip-quant` never requires Rust at runtime.

Usage
-----
>>> from hip_quant.smi import query, GpuMonitor
>>> query()  # one-shot
[{'index': 1, 'market_name': 'RX 9070 XT', 'gfx_version': 'gfx1201',
  'vram_used_mb': 820, 'vram_total_mb': 16304, 'gfx_util_percent': 88, ...}]

>>> monitor = GpuMonitor(interval=1.0)
>>> monitor.start()
>>> # ... training loop with wave_attn / fp8_linear ...
>>> for step in range(100):
...     out = hip_quant.wave_attn(q,k,v)
...     if step % 10 == 0:
...         print(monitor.latest())  # no subprocess block
>>> monitor.stop()
>>> print(monitor.summary())
>>> monitor.history()  # list of snapshots for plotting

Or headless HTTP for external dashboards:
>>> monitor.serve(port=8080)  # delegates to gpu-smi --serve

Reference: GPU-SMI repo amdsmi/include/amd_smi/amdsmi.h (MIT), ADL headers (MIT)
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import warnings
from typing import Any, Dict, List, Optional

_PKG_DIR = pathlib.Path(__file__).parent if pathlib.Path(__file__).parent.name != "hip_quant" else pathlib.Path(__file__).parent
# package root is repo root (pyproject packages hip_quant=".")
# binary locations: tools/gpu-smi.exe (vendored), or system PATH, or cargo build output
_BIN_LOCK = threading.Lock()

_CANDIDATES = [
    pathlib.Path(__file__).parent / "tools" / "gpu-smi.exe",
    pathlib.Path(__file__).parent / "tools" / "gpu-smi",
    pathlib.Path(__file__).parent / "gpu-smi.exe",
    pathlib.Path(__file__).parent / "gpu-smi",
    pathlib.Path(__file__).parent / "gpu-smi-src" / "target" / "release" / "gpu-smi.exe",
    pathlib.Path(__file__).parent / "gpu-smi-src" / "target" / "release" / "gpu-smi",
    pathlib.Path("C:/Users/armor/Desktop/hip_quant/tools/gpu-smi.exe"),
]


def _find_bin() -> Optional[pathlib.Path]:
    # explicit env override
    env = os.environ.get("HIP_QUANT_GPU_SMI_BIN") or os.environ.get("GPU_SMI_BIN")
    if env:
        p = pathlib.Path(env)
        if p.exists():
            return p
    for p in _CANDIDATES:
        if p.exists():
            return p
    # PATH fallback
    for name in ("gpu-smi.exe", "gpu-smi", "gpu_smi.exe"):
        import shutil

        found = shutil.which(name)
        if found:
            return pathlib.Path(found)
    return None


def _fallback_query() -> List[Dict[str, Any]]:
    """Fallback when gpu-smi binary missing: use torch.cuda + device_info."""
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    out = []
    try:
        # use device_info if available
        try:
            from .device_info import get_device_info  # type: ignore

            info = get_device_info()
            # get_device_info returns single; wrap
            gpus = [info] if not isinstance(info, list) else info
        except Exception:
            gpus = []

        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                free, total = torch.cuda.mem_get_info(i) if hasattr(torch.cuda, "mem_get_info") else (0, 0)
                used = (total - free) // (1024 * 1024) if total else 0
                total_mb = total // (1024 * 1024) if total else 0
                # try to enrich from device_info
                extra = {}
                for g in gpus:
                    # match by index or name
                    if getattr(g, "index", None) == i or getattr(g, "device_id", None) == props.major:
                        extra = g.__dict__ if hasattr(g, "__dict__") else {}
                        break
                arch = getattr(props, "gcnArchName", "") or getattr(props, "gcn_arch", "") or "unknown"
                out.append(
                    {
                        "index": i,
                        "market_name": props.name if hasattr(props, "name") else extra.get("market_name", f"cuda:{i}"),
                        "gfx_version": arch,
                        "vram_total_mb": total_mb,
                        "vram_used_mb": used,
                        "vram_type": extra.get("vram_type", "unknown"),
                        "temp_edge_c": None,
                        "temp_hotspot_c": None,
                        "gfx_clock_mhz": None,
                        "mem_clock_mhz": None,
                        "gfx_util_percent": None,
                        "power_w": None,
                        "pcie_width": extra.get("pcie_width", None),
                        "backend": "torch-fallback",
                    }
                )
            except Exception:
                continue
    except Exception:
        pass
    return out


def query(compact: bool = True, timeout: float = 3.0, retries: int = 1) -> List[Dict[str, Any]]:
    """One-shot query. Returns list[GpuInfo dict] (empty if no GPU).

    Tries gpu-smi binary (--compact one-liner) first; falls back to torch.
    `compact=True` uses `--compact` (single-line JSON, fastest for polling).
    Serialized with _BIN_LOCK — ADL dlopen is not thread-safe for concurrent calls.
    Retries once on access-violation (3221225477) which can happen under load.
    """
    bin_path = _find_bin()
    if bin_path is not None:
        for attempt in range(retries + 1):
            try:
                args = [str(bin_path), "--compact" if compact else "--json"]
                kwargs = {}
                if os.name == "nt":
                    kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                with _BIN_LOCK:
                    data = subprocess.check_output(args, timeout=timeout, **kwargs)
                return json.loads(data.decode("utf-8", errors="ignore"))
            except subprocess.CalledProcessError as exc:
                # access violation - retry after brief pause
                if attempt < retries and exc.returncode in (-1073741819, 3221225477):
                    time.sleep(0.12)
                    continue
                warnings.warn(f"gpu-smi query failed ({exc}), falling back to torch", RuntimeWarning, stacklevel=2)
                break
            except Exception as exc:
                warnings.warn(f"gpu-smi query failed ({exc}), falling back to torch", RuntimeWarning, stacklevel=2)
                break
    fb = _fallback_query()
    return fb if fb else []


def _fmt(v, suffix="", nil="-"):
    return f"{v}{suffix}" if v is not None else nil


def format_gpu(g: Dict[str, Any]) -> str:
    """One-line clean status — not a wall of JSON."""
    # e.g. RX 9070 XT | 842/16304 MB | 89% | 54C/70C | 3400/24MHz | 45.2W | hip+wmi+adl
    vram = f"{g.get('vram_used_mb',0)}/{g.get('vram_total_mb',0)} MB"
    util = _fmt(g.get("gfx_util_percent"), "%")
    temp = _fmt(g.get("temp_edge_c"), "C")
    if g.get("temp_hotspot_c") is not None:
        temp += f"/{g['temp_hotspot_c']:.0f}C"
    clk = _fmt(g.get("gfx_clock_mhz"), "MHz")
    if g.get("mem_clock_mhz") is not None:
        clk += f"/{g['mem_clock_mhz']}MHz"
    power = _fmt(g.get("power_w"), "W", nil="-")
    if power != "-":
        try:
            power = f"{float(power):.1f}W"
        except Exception:
            pass
    name = g.get("market_name", f"GPU{g.get('index',0)}")
    # trim long names
    if len(name) > 22:
        name = name[:19] + "..."
    return f"{name:<22} | {vram:<14} | {util:<4} | {temp:<9} | {clk:<14} | {power:<6} | {g.get('backend','')}"


def format_table(gpus: List[Dict[str, Any]]) -> str:
    hdr = f"{'GPU':<22} | {'VRAM':<14} | {'Util':<4} | {'Temp':<9} | {'Clocks':<14} | {'Power':<6} | Backend"
    sep = "-" * len(hdr)
    rows = [hdr, sep] + [format_gpu(g) for g in gpus]
    return "\n".join(rows)


def status(compact: bool = True) -> str:
    """Clean one-line per GPU — use while training: `print(smi.status())`."""
    return format_table(query(compact=compact))


def brief(index: int = 1) -> str:
    """Ultra-compact one-liner for a single GPU (defaults to dGPU 1 on desktop)."""
    g = query_one(index)
    return format_gpu(g) if g else "no GPU"


def query_one(index: int = 0, **kw) -> Optional[Dict[str, Any]]:
    """Convenience: query single GPU by index."""
    gpus = query(**kw)
    for g in gpus:
        if int(g.get("index", -1)) == int(index):
            return g
    return gpus[0] if gpus else None


class GpuMonitor:
    """Background sampler for training loops — polls `query()` without blocking forward.

    Example
    -------
    >>> monitor = GpuMonitor(interval=1.0)
    >>> monitor.start()
    >>> for step in range(steps):
    ...     loss = train_step()
    ...     if step % 20 == 0:
    ...         print(monitor.latest())
    >>> monitor.stop()
    >>> hist = monitor.history()
    """

    def __init__(self, interval: float = 1.0, compact: bool = True, max_history: int = 3600):
        self.interval = float(interval)
        self.compact = bool(compact)
        self.max_history = int(max_history)
        self._history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_ts: Optional[float] = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                snap = query(compact=self.compact, timeout=2.5, retries=1)
                # ensure non-empty; fallback may return [] if torch not ready yet
                if not snap:
                    snap = _fallback_query()
                ts = time.time()
                if snap:
                    with self._lock:
                        self._history.append({"ts": ts, "gpus": snap})
                        if len(self._history) > self.max_history:
                            self._history = self._history[-self.max_history :]
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self) -> "GpuMonitor":
        if self._thread and self._thread.is_alive():
            return self
        self._history.clear()
        self._stop.clear()
        self._start_ts = time.time()
        self._thread = threading.Thread(target=self._loop, name="hip-quant-smi", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> "GpuMonitor":
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self

    def latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._history[-1] if self._history else None

    def latest_str(self) -> str:
        """Clean one-liner for the most recent snapshot (no JSON wall)."""
        lat = self.latest()
        if not lat or not lat.get("gpus"):
            return "no data yet"
        return format_table(lat["gpus"])

    def history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history)

    def summary(self) -> Dict[str, Any]:
        """Peak / avg stats over history (dict). Use summary_str() for clean text."""
        hist = self.history()
        if not hist:
            return {}
        utils = [ (g.get("gfx_util_percent") or 0) for h in hist for g in h["gpus"] if g.get("gfx_util_percent") is not None ]
        powers = [ g.get("power_w") for h in hist for g in h["gpus"] if g.get("power_w") is not None ]
        temps = [ g.get("temp_edge_c") for h in hist for g in h["gpus"] if g.get("temp_edge_c") is not None ]
        vram_used = [ g.get("vram_used_mb") for h in hist for g in h["gpus"] if g.get("vram_used_mb") is not None ]
        def _stats(xs):
            return {"min": min(xs) if xs else None, "max": max(xs) if xs else None, "avg": sum(xs)/len(xs) if xs else None}
        return {
            "samples": len(hist),
            "duration_s": (hist[-1]["ts"] - hist[0]["ts"]) if len(hist) > 1 else 0,
            "util": _stats(utils),
            "power_w": _stats(powers) if powers else None,  # type: ignore[arg-type]
            "temp_edge_c": _stats(temps) if temps else None,  # type: ignore[arg-type]
            "vram_used_mb": _stats(vram_used) if vram_used else None,  # type: ignore[arg-type]
        }

    def summary_str(self) -> str:
        """Clean one-liner summary — `print(monitor.summary_str())`."""
        s = self.summary()
        if not s:
            return "no samples"
        def fmt(d, unit=""):
            if not d or d["avg"] is None:
                return "-"
            return f"{d['min']:.0f}-{d['max']:.0f} avg {d['avg']:.0f}{unit}"
        return (
            f"{s['samples']} samples {s['duration_s']:.0f}s | "
            f"VRAM {fmt(s['vram_used_mb'],'MB')} | "
            f"Util {fmt(s['util'],'%')} | "
            f"Temp {fmt(s['temp_edge_c'],'C')} | "
            f"Power {fmt(s['power_w'],'W')}"
        )

    def __enter__(self) -> "GpuMonitor":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def serve(self, port: int = 8080) -> subprocess.Popen:
        """Spawn gpu-smi --serve headless HTTP (for dashboards). Returns Popen."""
        bin_path = _find_bin()
        if bin_path is None:
            raise FileNotFoundError("gpu-smi binary not found — build with `cargo build --release` in gpu-smi-src/ or `build.ps1`")
        return subprocess.Popen([str(bin_path), "--serve", str(port)])


def main():
    """CLI entry `gpu-smi` / `hip-quant --smi`."""
    import argparse

    p = argparse.ArgumentParser(prog="gpu-smi", description="hip-quant GPU SMI — query AMD GPUs while training (via bundled gpu-smi)")
    p.add_argument("--json", action="store_true", help="pretty JSON")
    p.add_argument("--compact", action="store_true", help="compact JSON (for piping)")
    p.add_argument("--watch", nargs="?", const=1, type=int, metavar="SECS", help="realtime watch")
    p.add_argument("--serve", nargs="?", const=8080, type=int, metavar="PORT", help="headless HTTP")
    p.add_argument("--verbose", action="store_true", help="verbose table")
    args = p.parse_args()

    bin_path = _find_bin()
    if bin_path is None:
        print("gpu-smi binary not found, using fallback query:")
        print(json.dumps(query(), indent=2))
        return 0
    # delegate to binary for full features
    cmd = [str(bin_path)]
    if args.json:
        cmd.append("--json")
    if args.compact:
        cmd.append("--compact")
    if args.verbose:
        cmd.append("--verbose")
    if args.watch is not None:
        cmd += ["--watch", str(args.watch)]
    if args.serve is not None:
        cmd += ["--serve", str(args.serve)]
    if len(cmd) == 1:
        # default table
        pass
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        pass
    return 0


__all__ = [
    "query", "query_one", "brief", "status", "format_gpu", "format_table",
    "GpuMonitor", "_find_bin", "_fallback_query",
]
