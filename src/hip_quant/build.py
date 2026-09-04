import argparse
import glob
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

    if not rocm_bin:
        for env_name in ("HIP_QUANT_ROCM_BIN", "ROCM_PATH", "HIP_PATH", "ROCM_HOME"):
            val = os.environ.get(env_name)
            if val:
                p = Path(val if val.endswith("bin") else os.path.join(val, "bin"))
                if (p / "hipcc.exe").is_file():
                    rocm_bin = str(p)
                    break
    if not rocm_bin:
        candidates = sorted(glob.glob(r"C:\Program Files\AMD\ROCm\*\bin"), reverse=True)
        for c in candidates:
            if (Path(c) / "hipcc.exe").is_file():
                rocm_bin = str(c)
                break
    rocm_bin = rocm_bin or r"C:\Program Files\AMD\ROCm\7.1\bin"
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
    build(arch=args.arch, rocm_bin=args.rocm_bin, verbose=args.verbose)


if __name__ == "__main__":
    main()
