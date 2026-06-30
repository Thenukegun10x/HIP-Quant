import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def build(arch="all", rocm_bin=None, verbose=False):
    pkg_dir = Path(__file__).resolve().parent
    script = pkg_dir / "build.ps1"
    if not script.is_file():
        raise FileNotFoundError(f"build.ps1 not found at {script}")

    rocm_bin = rocm_bin or os.environ.get("HIP_QUANT_ROCM_BIN") or r"C:\Program Files\AMD\ROCm\7.1\bin"
    hipcc = Path(rocm_bin) / "hipcc.exe"
    if not hipcc.is_file():
        raise FileNotFoundError(
            f"hipcc.exe not found at {hipcc}. Set HIP_QUANT_ROCM_BIN or pass --rocm-bin."
        )

    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        raise FileNotFoundError("PowerShell was not found on PATH")

    env = os.environ.copy()
    env["HIP_QUANT_ROCM_BIN"] = str(rocm_bin)
    if arch:
        env["HIP_QUANT_ARCH"] = arch

    cmd = [powershell, "-ExecutionPolicy", "Bypass", "-File", str(script)]
    if verbose:
        print("Running:", " ".join(cmd))
        print("Working directory:", pkg_dir)
    subprocess.check_call(cmd, cwd=str(pkg_dir), env=env)
    dll = pkg_dir / "hip_quantize.dll"
    if not dll.is_file():
        raise RuntimeError(f"Build completed but {dll} was not created")
    return dll


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build hip_quantize.dll with hipcc.")
    parser.add_argument("--arch", default=os.environ.get("HIP_QUANT_ARCH", "all"), help="HIP offload arch, comma list, or all")
    parser.add_argument("--rocm-bin", default=os.environ.get("HIP_QUANT_ROCM_BIN"), help="Path to ROCm bin directory")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    dll = build(arch=args.arch, rocm_bin=args.rocm_bin, verbose=args.verbose)
    print(f"Built {dll}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
