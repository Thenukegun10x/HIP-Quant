# setup_torch.py
#
# Build script for the hip_quant PyTorch C++ extension.
#
# Usage:
#   python setup_torch.py build_ext --inplace
#
# On ROCm PyTorch, CUDAExtension is the correct class to use even though
# HIP/ROCm is the actual backend.  PyTorch's build system maps it to hipcc
# automatically when a ROCm-enabled PyTorch is detected.
#
# The --offload-arch flags target RDNA4 gfx1200/gfx1201 ASICs.

import glob
import os
from pathlib import Path


def _short_path(path):
    if os.name != "nt":
        return str(path)
    path = str(path)
    if not os.path.exists(path):
        return path
    try:
        import ctypes

        size = ctypes.windll.kernel32.GetShortPathNameW(path, None, 0)
        if size == 0:
            return path
        buf = ctypes.create_unicode_buffer(size)
        ctypes.windll.kernel32.GetShortPathNameW(path, buf, size)
        return buf.value or path
    except Exception:
        return path


def _prepend_path(path):
    path = _short_path(path)
    os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")


def _configure_windows_toolchain():
    if os.name != "nt":
        return

    rocm_home = Path(os.environ.get("HIP_QUANT_ROCM_HOME", r"C:\Program Files\AMD\ROCm\7.1"))
    if rocm_home.exists():
        rocm_home_short = _short_path(rocm_home)
        hip_clang_path = rocm_home / "lib" / "llvm" / "bin"
        os.environ["ROCM_HOME"] = rocm_home_short
        os.environ["HIP_PATH"] = rocm_home_short
        os.environ["HIP_CLANG_PATH"] = _short_path(hip_clang_path if hip_clang_path.exists() else rocm_home / "bin")
        _prepend_path(rocm_home / "bin")
        if hip_clang_path.exists():
            _prepend_path(hip_clang_path)

    if "CC" not in os.environ or "CXX" not in os.environ:
        candidates = glob.glob(
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\*"
            r"\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"
        )
        if candidates:
            cl = Path(sorted(candidates)[-1])
            cl_short = _short_path(cl)
            os.environ.setdefault("CC", cl_short)
            os.environ.setdefault("CXX", cl_short)
            _prepend_path(cl.parent)

    # link.exe invokes rc.exe to embed the extension manifest. vcvars64.bat
    # does not always add a Windows SDK bin directory in non-interactive
    # shells, which otherwise makes an otherwise successful HIP build fail at
    # the final link step with LNK1158.
    resource_compilers = glob.glob(
        r"C:\Program Files (x86)\Windows Kits\10\bin\*\x64\rc.exe"
    )
    if resource_compilers:
        _prepend_path(Path(sorted(resource_compilers)[-1]).parent)

    if os.environ.get("VSCMD_VER"):
        os.environ.setdefault("DISTUTILS_USE_SDK", "1")

    os.environ.setdefault("MAX_JOBS", "1")


_configure_windows_toolchain()

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths

# Convert all torch include paths to Windows 8.3 short paths to avoid space-related build failures
torch_includes = [_short_path(p) for p in include_paths()]

ext = CUDAExtension(
    # The extension will be importable as  hip_quant._C
    name="hip_quant._C",
    sources=[
        "torch_ext/pytorch_bindings.cpp",
        "torch_ext/current_stream.hip",
        "torch_ext/fp8_quant_kernels.hip",
        "torch_ext/mxfp8_kernels.hip",
        "torch_ext/fp8_linear_kernels.hip",
        "torch_ext/fp8_linear_kernels_v2.hip",
        "torch_ext/q_to_fp8_kernels.hip",
        "torch_ext/mxfp4_to_fp8_kernels.hip",
        "torch_ext/hq2_linear_kernels.hip",
        "torch_ext/hq3_linear_kernels.hip",
        "torch_ext/hq8_g128_linear_kernels.hip",
        "torch_ext/wave_attn.hip",
        "torch_ext/wave_attn_backward.hip",
        "torch_ext/wave_attn_diag.hip",
    ],
    extra_compile_args={
        # Host (clang++ / g++) flags
        "cxx": ["-O3", "-DHIP_QUANT_ENABLE_NONTEMPORAL=1"],
        # Device (hipcc) flags; nvcc key is used by PyTorch even on ROCm
        "nvcc": [
            "-O3",
            "-mno-wavefrontsize64",
            # One-pass FP8 conversion has no in-kernel reuse.  Clang lowers
            # the guarded helpers to AMDGPU non-temporal memory operations.
            "-DHIP_QUANT_ENABLE_NONTEMPORAL=1",
            "--offload-arch=gfx1200",
            "--offload-arch=gfx1201",
            # Allow device code to include project headers via relative paths
            "-I.",
        ] + [f"-I{p}" for p in torch_includes],
    },
    # Ensure headers shipped with the package are visible to hipcc
    include_dirs=["."] + torch_includes,
    # The native MXFP4 binding resolves hipBLASLt dynamically on Windows to
    # avoid requiring an MSVC import library. POSIX builds need libdl for the
    # equivalent dlopen/dlsym path.
    extra_link_args=[] if os.name == "nt" else ["-ldl"],
)

setup(
    name="hip_quant_torch",
    version="0.13.0",
    description="PyTorch FP8 extension for hip_quant (AMD ROCm / HIP)",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)
