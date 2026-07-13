"""
tests/torch/test_fp8_v2.py

Correctness + microbenchmark for the V2 cooperative LDS-staged FP8 GEMM kernel.

Run with:
    pytest tests/torch/test_fp8_v2.py -v
"""
import math
import pytest
import torch

torch_available = True
try:
    import torch
except ImportError:
    torch_available = False

extension_available = False
if torch_available:
    try:
        from hip_quant import _C
        if not hasattr(_C, "fp8_linear_forward_v2_input_weight"):
            pass  # v2 not built
        else:
            extension_available = True
    except ImportError:
        pass

pytestmark = pytest.mark.skipif(
    not torch_available or not extension_available,
    reason="Requires PyTorch ROCm + hip_quant._C with v2 kernel",
)


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA/HIP GPU available")
    return torch.device("cuda")


@pytest.fixture(scope="module")
def quant_funcs():
    from hip_quant.torch_api import quantize_e4m3
    return quantize_e4m3


# ---- correctness ----
class TestV2Correctness:
    def test_v2_matches_v1_small(self, device, quant_funcs):
        from hip_quant.torch_api import (
            fp8_linear_forward_fp8_input_weight,
            fp8_linear_forward_v2_input_weight,
        )
        torch.manual_seed(42)
        M, N, K = 64, 128, 256
        inp = torch.randn(M, K, device=device, dtype=torch.float32)
        weight = torch.randn(N, K, device=device, dtype=torch.float32)
        input_fp8 = quant_funcs(inp)
        weight_fp8 = quant_funcs(weight)

        out_v1 = fp8_linear_forward_fp8_input_weight(
            input_fp8, weight_fp8, inp,
            weight_inv_scale=1.0, input_scale=1.0, bias=None,
        )
        out_v2 = fp8_linear_forward_v2_input_weight(
            input_fp8, weight_fp8, inp,
            weight_inv_scale=1.0, input_scale=1.0, bias=None,
        )
        diff = (out_v1.float() - out_v2.float()).abs().max().item()
        assert diff < 1e-3, f"V1 vs V2 max diff={diff}"

    def test_v2_matches_v1_large(self, device, quant_funcs):
        from hip_quant.torch_api import (
            fp8_linear_forward_fp8_input_weight,
            fp8_linear_forward_v2_input_weight,
        )
        torch.manual_seed(123)
        M, N, K = 128, 512, 1024
        inp = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        weight = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        input_fp8 = quant_funcs(inp)
        weight_fp8 = quant_funcs(weight)

        out_v1 = fp8_linear_forward_fp8_input_weight(
            input_fp8, weight_fp8, inp,
            weight_inv_scale=1.0, input_scale=1.0, bias=None,
        )
        out_v2 = fp8_linear_forward_v2_input_weight(
            input_fp8, weight_fp8, inp,
            weight_inv_scale=1.0, input_scale=1.0, bias=None,
        )
        diff = (out_v1.float() - out_v2.float()).abs().max().item()
        assert diff < 1e-3, f"V1 vs V2 max diff={diff}"

    def test_v2_matches_decoded_fp8_reference(self, device, quant_funcs):
        from hip_quant.torch_api import (
            dequantize_e4m3,
            fp8_linear_forward_v2_input_weight,
        )
        torch.manual_seed(99)
        M, N, K = 32, 64, 128
        inp = torch.randn(M, K, device=device, dtype=torch.float32)
        weight = torch.randn(N, K, device=device, dtype=torch.float32)
        input_fp8 = quant_funcs(inp)
        weight_fp8 = quant_funcs(weight)

        out_v2 = fp8_linear_forward_v2_input_weight(
            input_fp8, weight_fp8, inp,
            weight_inv_scale=1.0, input_scale=1.0, bias=None,
        )
        # WMMA consumes the quantized bytes, so its correctness reference is
        # the GEMM of their decoded FP8 values—not the original FP32 tensors.
        ref = dequantize_e4m3(input_fp8).matmul(dequantize_e4m3(weight_fp8).T)
        torch.testing.assert_close(out_v2.float(), ref.float(), rtol=2e-3, atol=2e-3)

    def test_v2_with_bias(self, device, quant_funcs):
        from hip_quant.torch_api import fp8_linear_forward_v2_input_weight
        torch.manual_seed(77)
        M, N, K = 16, 32, 64
        inp = torch.randn(M, K, device=device, dtype=torch.float32)
        weight = torch.randn(N, K, device=device, dtype=torch.float32)
        bias = torch.randn(N, device=device, dtype=torch.float32)
        input_fp8 = quant_funcs(inp)
        weight_fp8 = quant_funcs(weight)

        out = fp8_linear_forward_v2_input_weight(
            input_fp8, weight_fp8, inp,
            weight_inv_scale=1.0, input_scale=1.0, bias=bias,
        )
        assert out.shape == (M, N)
        assert out.dtype == inp.dtype

    @pytest.mark.parametrize("M", [17, 31, 33])
    def test_v2_matches_v1_tail_tiles(self, device, quant_funcs, M):
        from hip_quant.torch_api import (
            fp8_linear_forward_fp8_input_weight,
            fp8_linear_forward_v2_input_weight,
        )
        torch.manual_seed(M)
        N, K = 37, 133
        inp = torch.randn(M, K, device=device, dtype=torch.float32)
        weight = torch.randn(N, K, device=device, dtype=torch.float32)
        input_fp8 = quant_funcs(inp)
        weight_fp8 = quant_funcs(weight)

        out_v1 = fp8_linear_forward_fp8_input_weight(
            input_fp8, weight_fp8, inp, 1.0, 1.0, bias=None)
        out_v2 = fp8_linear_forward_v2_input_weight(
            input_fp8, weight_fp8, inp, 1.0, 1.0, bias=None)

        torch.testing.assert_close(out_v2, out_v1, rtol=0.0, atol=1e-3)


# ---- microbenchmark ----
def _time_cuda(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _gflops(M, N, K, ms):
    return 2.0 * M * N * K / (ms * 1e6)


class TestV2Benchmark:
    def test_bench_v1_vs_v2_vs_hipblaslt(self, device, quant_funcs):
        from hip_quant.torch_api import (
            fp8_linear_forward_fp8_input_weight,
            fp8_linear_forward_v2_input_weight,
        )
        torch.manual_seed(42)

        dims = [(32, 2048, 2048), (128, 2048, 2048), (32, 4096, 4096), (128, 4096, 4096)]
        print()
        for M, N, K in dims:
            inp = torch.randn(M, K, device=device, dtype=torch.bfloat16)
            weight = torch.randn(N, K, device=device, dtype=torch.bfloat16)
            input_fp8 = quant_funcs(inp)
            weight_fp8 = quant_funcs(weight)

            t_v1 = _time_cuda(lambda: fp8_linear_forward_fp8_input_weight(
                input_fp8, weight_fp8, inp, 1.0, 1.0, bias=None), warmup=10, iters=50)
            t_v2 = _time_cuda(lambda: fp8_linear_forward_v2_input_weight(
                input_fp8, weight_fp8, inp, 1.0, 1.0, bias=None), warmup=10, iters=50)
            ratio = t_v1 / t_v2 if t_v2 > 0 else float('inf')

            # hipBLASLt if available
            if hasattr(torch, "_scaled_mm") and hasattr(torch, "float8_e4m3fn"):
                inp_f8 = inp.float().to(torch.float8_e4m3fn)
                w_f8_t = weight.float().to(torch.float8_e4m3fn).contiguous().t()
                s_a = torch.ones((), device=device, dtype=torch.float32)
                s_b = torch.ones((), device=device, dtype=torch.float32)
                t_blas = _time_cuda(lambda: torch._scaled_mm(inp_f8, w_f8_t, s_a, s_b, out_dtype=torch.bfloat16), warmup=10, iters=50)
            else:
                t_blas = None

            print(f"M={M:4d} N={N:5d} K={K:5d} | v1={t_v1:.3f}ms v2={t_v2:.3f}ms", end="")
            if t_blas is not None:
                print(f" blas={t_blas:.3f}ms", end="")
            print(f" | v2/v1={ratio:.2f}x")
            assert t_v2 > 0, "V2 kernel appears to hang or fail"

    def test_bench_k_scan(self, device, quant_funcs):
        """Vary K to see how the K_TILES_PER_STAGE=8 sweet spot performs."""
        from hip_quant.torch_api import (
            fp8_linear_forward_fp8_input_weight,
            fp8_linear_forward_v2_input_weight,
        )
        torch.manual_seed(42)
        M, N = 32, 2048
        print()
        for K in [256, 512, 1024, 2048, 4096, 8192]:
            inp = torch.randn(M, K, device=device, dtype=torch.bfloat16)
            weight = torch.randn(N, K, device=device, dtype=torch.bfloat16)
            input_fp8 = quant_funcs(inp)
            weight_fp8 = quant_funcs(weight)

            t_v1 = _time_cuda(lambda: fp8_linear_forward_fp8_input_weight(
                input_fp8, weight_fp8, inp, 1.0, 1.0, bias=None), warmup=10, iters=30)
            t_v2 = _time_cuda(lambda: fp8_linear_forward_v2_input_weight(
                input_fp8, weight_fp8, inp, 1.0, 1.0, bias=None), warmup=10, iters=30)
            ratio = t_v1 / t_v2 if t_v2 > 0 else float('inf')
            print(f"K={K:5d} | v1={t_v1:.3f}ms v2={t_v2:.3f}ms | v2/v1={ratio:.2f}x v1_gflops={_gflops(M,N,K,t_v1):.0f} v2_gflops={_gflops(M,N,K,t_v2):.0f}")
            assert t_v2 > 0

    def test_bench_v2_vs_native_packed(self, device, quant_funcs):
        from hip_quant.torch_api import (
            fp8_linear_forward_fp8_input_weight_packed,
            fp8_linear_forward_v2_input_weight,
            pack_fp8_weight_for_wmma,
        )
        torch.manual_seed(42)

        dims = [(32, 2048, 2048), (128, 2048, 2048), (32, 4096, 4096), (128, 4096, 4096)]
        print()
        for M, N, K in dims:
            inp = torch.randn(M, K, device=device, dtype=torch.bfloat16)
            weight = torch.randn(N, K, device=device, dtype=torch.bfloat16)
            input_fp8 = quant_funcs(inp)
            weight_fp8 = quant_funcs(weight)
            weight_packed = pack_fp8_weight_for_wmma(weight_fp8)

            t_v2 = _time_cuda(lambda: fp8_linear_forward_v2_input_weight(
                input_fp8, weight_fp8, inp, 1.0, 1.0, bias=None), warmup=10, iters=50)
            t_packed = _time_cuda(lambda: fp8_linear_forward_fp8_input_weight_packed(
                input_fp8, weight_packed, inp, N, 1.0, 1.0, bias=None), warmup=10, iters=50)

            print(f"M={M:4d} N={N:5d} K={K:5d} | v2={t_v2:.3f}ms packed={t_packed:.3f}ms | packed/v2={t_v2 / t_packed:.2f}x")
            assert t_packed > 0
