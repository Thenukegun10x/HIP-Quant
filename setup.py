import glob
import os
from pathlib import Path

from setuptools import setup

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    _bdist_wheel = None


if _bdist_wheel is not None:
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False
else:
    bdist_wheel = None


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
        os.environ["ROCM_HOME"] = rocm_home_short
        os.environ["HIP_PATH"] = rocm_home_short
        os.environ["HIP_CLANG_PATH"] = _short_path(rocm_home / "bin")
        _prepend_path(rocm_home / "bin")

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

    if os.environ.get("VSCMD_VER"):
        os.environ.setdefault("DISTUTILS_USE_SDK", "1")

    os.environ.setdefault("MAX_JOBS", "1")


def _torch_extension_config():
    base_cmdclass = {}
    if bdist_wheel is not None:
        base_cmdclass["bdist_wheel"] = bdist_wheel

    if os.environ.get("HIP_QUANT_BUILD_TORCH_EXT") != "1":
        return [], base_cmdclass

    _configure_windows_toolchain()

    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    ext = CUDAExtension(
        name="hip_quant._C",
        sources=[
            "torch_ext/pytorch_bindings.cpp",
            "torch_ext/fp8_quant_kernels.hip",
            "torch_ext/fp8_linear_kernels.hip",
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "-mno-wavefrontsize64",
                "--offload-arch=gfx1200",
                "--offload-arch=gfx1201",
                "-I.",
            ],
        },
        include_dirs=["."],
    )
    base_cmdclass["build_ext"] = BuildExtension
    return [ext], base_cmdclass


ext_modules, cmdclass = _torch_extension_config()

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
)
