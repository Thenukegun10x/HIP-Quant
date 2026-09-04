"""Serialized, resource-guarded launcher for llama.cpp binaries.

This exists because of a real incident: two `llama-cli` processes were left
alive simultaneously, each holding ~11 GiB of VRAM on a 15.9 GiB card and each
mmapping the same 23.8 GB F16 model against ~13.7 GiB of free host RAM.  The
page cache thrashed, Windows grew the pagefile on an already 90%-full C:, and
the machine hard-froze badly enough that the volume needed repair on reboot.

The rules enforced here, in order of importance:

1. Exactly one llama.cpp process at a time.  Any survivor is killed and its
   death confirmed before a new one starts.
2. Never launch without free-RAM and free-VRAM headroom.
3. Every launch has a hard timeout and is killed on any exit path, including
   KeyboardInterrupt and timeout.
4. Never launch anything that can block on stdin.  stdin is closed, and
   interactive flags are rejected outright.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

LLAMA_IMAGES = ("llama-cli.exe", "llama-perplexity.exe", "llama-imatrix.exe",
                "llama-server.exe", "llama-bench.exe", "llama-quantize.exe")

# Flags that make a llama.cpp binary wait on stdin.  A blocked process holds
# VRAM and page cache indefinitely, which is precisely what caused the freeze.
FORBIDDEN_FLAGS = {"-i", "--interactive", "-cnv", "--conversation", "-ins", "--interactive-first"}

MIN_FREE_RAM_GIB = 6.0
MIN_FREE_VRAM_GIB = 13.0


def _run(cmd: list[str], timeout: int = 30) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout
    except Exception:
        return ""


def running_llama() -> list[str]:
    alive = []
    for image in LLAMA_IMAGES:
        out = _run(["tasklist", "/FI", f"IMAGENAME eq {image}"])
        if image.lower() in out.lower():
            alive.append(image)
    return alive


def kill_all_llama(verbose: bool = True) -> None:
    """Kill every llama.cpp process and block until they are actually gone."""
    for image in LLAMA_IMAGES:
        _run(["taskkill", "/F", "/IM", image])
    for _ in range(40):  # up to ~20 s
        alive = running_llama()
        if not alive:
            if verbose:
                print("[safe_run] no llama processes running", flush=True)
            return
        time.sleep(0.5)
    raise RuntimeError(f"could not kill lingering llama processes: {running_llama()}")


def free_ram_gib() -> float:
    out = _run(["wmic", "OS", "get", "FreePhysicalMemory", "/format:list"])
    for line in out.splitlines():
        if "FreePhysicalMemory" in line:
            try:
                return int(line.split("=")[1].strip()) / 1024 / 1024
            except (IndexError, ValueError):
                pass
    return -1.0


def free_vram_gib() -> float:
    """Query VRAM without importing torch into this process' address space."""
    probe = (
        "import torch,sys\n"
        "free,_=torch.cuda.mem_get_info() if torch.cuda.is_available() else (0,0)\n"
        "sys.stdout.write(str(free/1024**3))\n"
    )
    python = Path(__file__).resolve().parents[2] / ".venv" / "Scripts" / "python.exe"
    try:
        out = subprocess.run([str(python), "-c", probe], capture_output=True, text=True, timeout=120)
        return float(out.stdout.strip().splitlines()[-1])
    except Exception:
        return -1.0


def preflight(require_vram: bool = True) -> None:
    kill_all_llama()

    ram = free_ram_gib()
    print(f"[safe_run] free RAM: {ram:.1f} GiB (floor {MIN_FREE_RAM_GIB})", flush=True)
    if 0 <= ram < MIN_FREE_RAM_GIB:
        raise RuntimeError(f"only {ram:.1f} GiB RAM free; refusing to launch")

    if require_vram:
        vram = free_vram_gib()
        print(f"[safe_run] free VRAM: {vram:.1f} GiB (floor {MIN_FREE_VRAM_GIB})", flush=True)
        if 0 <= vram < MIN_FREE_VRAM_GIB:
            raise RuntimeError(f"only {vram:.1f} GiB VRAM free; a previous run may not have released it")


def launch(cmd: list[str], log_path: str | Path, timeout_s: int, require_vram: bool = True) -> int:
    """Run one llama.cpp command to completion under every guard above."""
    for arg in cmd:
        if arg in FORBIDDEN_FLAGS:
            raise ValueError(f"refusing to launch with interactive flag {arg!r}: it can block on stdin forever")

    preflight(require_vram=require_vram)

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[safe_run] launching (timeout {timeout_s}s): {' '.join(shlex.quote(c) for c in cmd)}", flush=True)

    started = time.time()
    process = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,  # cannot block waiting for input
            )
            code = process.wait(timeout=timeout_s)
        print(f"[safe_run] exited {code} after {time.time() - started:.0f}s", flush=True)
        return code
    except subprocess.TimeoutExpired:
        print(f"[safe_run] TIMEOUT after {timeout_s}s - killing", flush=True)
        return -1
    finally:
        # Every exit path, including timeout and Ctrl-C, ends with nothing alive.
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=30)
            except Exception:
                pass
        kill_all_llama(verbose=False)
        leftover = running_llama()
        if leftover:
            print(f"[safe_run] WARNING: still alive after cleanup: {leftover}", flush=True)
        else:
            print("[safe_run] cleanup verified: no llama processes remain", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--no-vram-check", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not args.command:
        parser.error("no command given")
    cmd = args.command[1:] if args.command[0] == "--" else args.command
    return launch(cmd, args.log, args.timeout, require_vram=not args.no_vram_check)


if __name__ == "__main__":
    sys.exit(main())
