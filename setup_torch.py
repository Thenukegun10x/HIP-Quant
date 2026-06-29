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
# The --offload-arch flag is passed to hipcc to target gfx1201 (RX 9070 XT).
# Adjust this if you target a different GPU architecture.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext = CUDAExtension(
    # The extension will be importable as  hip_quant._C
    name="hip_quant._C",
    sources=[
        "torch_ext/pytorch_bindings.cpp",
        "torch_ext/fp8_quant_kernels.hip",
        "torch_ext/fp8_linear_kernels.hip",
    ],
    extra_compile_args={
        # Host (clang++ / g++) flags
        "cxx": ["-O3"],
        # Device (hipcc) flags; nvcc key is used by PyTorch even on ROCm
        "nvcc": [
            "-O3",
            "--offload-arch=gfx1201",
            # Allow device code to include project headers via relative paths
            "-I.",
        ],
    },
    # Ensure headers shipped with the package are visible to hipcc
    include_dirs=["."],
)

setup(
    name="hip_quant_torch",
    version="0.1.0",
    description="PyTorch FP8 extension for hip_quant (AMD ROCm / HIP)",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)
