// torch_ext/pytorch_bindings.cpp
//
// PyTorch C++ extension bindings for hip_quant FP8 operations.
// Compiled with torch.utils.cpp_extension.CUDAExtension (ROCm/HIP mode).
//
// Exposed Python functions (via pybind11 / TORCH_EXTENSION_NAME):
//   Phase 1 — element-wise FP8 quant/dequant:
//     quantize_e4m3(Tensor) -> Tensor[uint8]
//     quantize_e5m2(Tensor) -> Tensor[uint8]
//     dequantize_e4m3(Tensor) -> Tensor[float32]
//     dequantize_e5m2(Tensor) -> Tensor[float32]
//
//   Phase 4 — FP8 linear GEMM:
//     fp8_linear_forward(input, weight, bias?) -> Tensor[float32]
//     fp8_linear_backward_input(grad_output, weight) -> Tensor[float32]
//     fp8_linear_backward_weight(grad_output, input) -> Tensor[float32]
//
// The actual kernel launches live in fp8_quant_kernels.hip and
// fp8_linear_kernels.hip so this file only does argument validation and
// tensor allocation.

#include <torch/all.h>
#include <torch/csrc/utils/pybind.h>
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

#ifdef TORCH_CHECK
#undef TORCH_CHECK
#endif
#define TORCH_CHECK(cond, ...) \
    do { if (!(cond)) throw std::runtime_error("hip_quant validation failed: " #__VA_ARGS__); } while (0)

#ifdef _MSC_VER
// PyTorch 2.9.1+ROCm Windows headers reference this inherited constructor,
// but the wheel's c10.lib only exports the base c10::Error overload.
#pragma comment(linker, "/alternatename:__imp_??0ValueError@c10@@QEAA@USourceLocation@1@V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z=__imp_??0Error@c10@@QEAA@USourceLocation@1@V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z")
#endif

// Forward declarations for kernel launchers defined in the .hip files.
extern "C" {
void launch_quant_e4m3(const void* src, uint8_t* dst, int64_t numel,
                       int dtype, hipStream_t stream);
void launch_quant_e5m2(const void* src, uint8_t* dst, int64_t numel,
                       int dtype, hipStream_t stream);
void launch_dequant_e4m3(const uint8_t* src, float* dst, int64_t numel,
                         hipStream_t stream);
void launch_dequant_e5m2(const uint8_t* src, float* dst, int64_t numel,
                         hipStream_t stream);

void launch_fp8_linear_forward(
    const void* A, const void* B, void* C,
    int M, int N, int K, int a_dtype, int b_dtype, int c_dtype,
    hipStream_t stream);
void launch_fp8_linear_forward_scaled(
    const void* A, const void* B, void* C,
    int M, int N, int K, float input_scale, float weight_scale,
    int a_dtype, int b_dtype, int c_dtype,
    hipStream_t stream);
void launch_fp8_linear_forward_fp8_weight(
    const void* A, const uint8_t* B_fp8, void* C,
    int M, int N, int K, float input_scale, float weight_inv_scale,
    int a_dtype, int c_dtype,
    hipStream_t stream);
void launch_fp8_linear_backward_input(
    const void* grad_output, const void* weight, void* grad_input,
    int M, int N, int K, int grad_output_dtype, int weight_dtype,
    int grad_input_dtype, hipStream_t stream);
void launch_fp8_linear_backward_input_scaled(
    const void* grad_output, const void* weight, void* grad_input,
    int M, int N, int K, float weight_scale, int grad_output_dtype,
    int weight_dtype, int grad_input_dtype, hipStream_t stream);
void launch_fp8_linear_backward_weight(
    const void* grad_output, const void* input, void* grad_weight,
    int M, int N, int K, int grad_output_dtype, int input_dtype,
    int grad_weight_dtype, hipStream_t stream);
void launch_fp8_linear_backward_weight_scaled(
    const void* grad_output, const void* input, void* grad_weight,
    int M, int N, int K, float input_scale, int grad_output_dtype,
    int input_dtype, int grad_weight_dtype, hipStream_t stream);
} // extern "C"

constexpr int HIP_QUANT_DTYPE_F32  = 0;
constexpr int HIP_QUANT_DTYPE_F16  = 1;
constexpr int HIP_QUANT_DTYPE_BF16 = 2;

// ---------------------------------------------------------------------------
// Helper: get current HIP stream from ATen
// ---------------------------------------------------------------------------
static hipStream_t current_stream() {
    return nullptr;
}

static inline int float_dtype_code(c10::ScalarType dtype, const char* name) {
    if (dtype == torch::kFloat32) return HIP_QUANT_DTYPE_F32;
    if (dtype == torch::kFloat16) return HIP_QUANT_DTYPE_F16;
    if (dtype == torch::kBFloat16) return HIP_QUANT_DTYPE_BF16;
    TORCH_CHECK(false, name, " must be float32, float16, or bfloat16");
    return HIP_QUANT_DTYPE_F32;
}

static inline int float_dtype_code(const torch::Tensor& tensor, const char* name) {
    return float_dtype_code(tensor.scalar_type(), name);
}

// ---------------------------------------------------------------------------
// Helper: safe int64 → int narrowing with TORCH_CHECK.
// Bug 2: without this, large tensors silently overflow the int arithmetic
// inside the GEMM kernels (row * K + k wraps at ~2^31).
// ---------------------------------------------------------------------------
static inline int checked_int(int64_t v, const char* name) {
    TORCH_CHECK(v >= 0 && v <= (int64_t)INT_MAX,
                name, " dimension ", v,
                " exceeds INT_MAX; use smaller tensors with the FP8 GEMM kernel.");
    return (int)v;
}

// ---------------------------------------------------------------------------
// Helper: validate that a GEMM grid dimension fits within hardware limits.
// Bug 3: gfx1201 supports gridDim.y up to 65535.
// We use 65535 as the conservative limit for both x and y.
// ---------------------------------------------------------------------------
static inline void check_grid_dim(int64_t dim, int tile, const char* axis) {
    int64_t blocks = (dim + tile - 1) / tile;
    TORCH_CHECK(blocks <= 65535,
                "fp8_linear: grid dimension on ", axis, " (", blocks,
                " blocks) exceeds HW limit of 65535. "
                "Reduce the corresponding tensor dimension or increase TILE size.");
}

static inline void check_gfx12_fp8_wmma_runtime(const char* op_name) {
    const char* disable_wmma = std::getenv("HIP_QUANT_DISABLE_WMMA");
    TORCH_CHECK(!(disable_wmma != nullptr && (
                    std::strcmp(disable_wmma, "1") == 0 || std::strcmp(disable_wmma, "true") == 0 ||
                    std::strcmp(disable_wmma, "yes") == 0 || std::strcmp(disable_wmma, "on") == 0)),
                op_name, ": disabled by HIP_QUANT_DISABLE_WMMA");

    const char* enable_wmma = std::getenv("HIP_QUANT_ENABLE_GFX12_WMMA");
    TORCH_CHECK(enable_wmma != nullptr && (
                    std::strcmp(enable_wmma, "1") == 0 || std::strcmp(enable_wmma, "true") == 0 ||
                    std::strcmp(enable_wmma, "yes") == 0 || std::strcmp(enable_wmma, "on") == 0),
                op_name,
                ": disabled by default because unstable FP8/BF8 WMMA kernels can hang or reset the GPU. "
                "Set HIP_QUANT_ENABLE_GFX12_WMMA=1 only for controlled testing on ROCm 7.2+ gfx12 systems.");

    int device = 0;
    hipError_t err = hipGetDevice(&device);
    TORCH_CHECK(err == hipSuccess, op_name, ": hipGetDevice failed");

    hipDeviceProp_t props;
    err = hipGetDeviceProperties(&props, device);
    TORCH_CHECK(err == hipSuccess, op_name, ": hipGetDeviceProperties failed");
    TORCH_CHECK(strstr(props.gcnArchName, "gfx12") != nullptr,
                op_name, ": hip_quant FP8/BF8 WMMA kernels use gfx12/RDNA4 w32 intrinsics; current arch is ",
                props.gcnArchName,
                ". CDNA may support FP8/BF16 through MFMA/rocBLASLt paths, but not this RDNA4-specific kernel.");

    int runtime_version = 0;
    hipRuntimeGetVersion(&runtime_version);
    TORCH_CHECK(runtime_version == 0 || runtime_version >= 70200000,
                op_name, ": ROCm/HIP 7.2+ is required for gfx12 FP8 WMMA; current runtime is ",
                runtime_version,
                ". ROCm 7.1 and older can hang or zero GPU memory.");
}

// ---------------------------------------------------------------------------
// Phase 1 — quantize_e4m3
// ---------------------------------------------------------------------------
torch::Tensor quantize_e4m3(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(),       "quantize_e4m3: input must be a HIP/CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "quantize_e4m3: input must be contiguous");
    int input_dtype = float_dtype_code(input, "quantize_e4m3: input");

    auto output = torch::empty(input.sizes(),
                               input.options().dtype(torch::kUInt8));
    int64_t numel = input.numel();
    launch_quant_e4m3(
        input.data_ptr(),
        output.data_ptr<uint8_t>(),
        numel,
        input_dtype,
        current_stream()
    );
    return output;
}

// ---------------------------------------------------------------------------
// Phase 1 — quantize_e5m2
// ---------------------------------------------------------------------------
torch::Tensor quantize_e5m2(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(),       "quantize_e5m2: input must be a HIP/CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "quantize_e5m2: input must be contiguous");
    int input_dtype = float_dtype_code(input, "quantize_e5m2: input");

    auto output = torch::empty(input.sizes(),
                               input.options().dtype(torch::kUInt8));
    int64_t numel = input.numel();
    launch_quant_e5m2(
        input.data_ptr(),
        output.data_ptr<uint8_t>(),
        numel,
        input_dtype,
        current_stream()
    );
    return output;
}

// ---------------------------------------------------------------------------
// Phase 1 — dequantize_e4m3
// ---------------------------------------------------------------------------
torch::Tensor dequantize_e4m3(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(),       "dequantize_e4m3: input must be a HIP/CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "dequantize_e4m3: input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kUInt8,
                "dequantize_e4m3: input must be uint8");

    auto output = torch::empty(input.sizes(),
                               input.options().dtype(torch::kFloat32));
    int64_t numel = input.numel();
    launch_dequant_e4m3(
        input.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        numel,
        current_stream()
    );
    return output;
}

// ---------------------------------------------------------------------------
// Phase 1 — dequantize_e5m2
// ---------------------------------------------------------------------------
torch::Tensor dequantize_e5m2(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(),       "dequantize_e5m2: input must be a HIP/CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "dequantize_e5m2: input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kUInt8,
                "dequantize_e5m2: input must be uint8");

    auto output = torch::empty(input.sizes(),
                               input.options().dtype(torch::kFloat32));
    int64_t numel = input.numel();
    launch_dequant_e5m2(
        input.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        numel,
        current_stream()
    );
    return output;
}

// ---------------------------------------------------------------------------
// Phase 4 — fp8_linear_forward
// input : [M, K] float32
// weight: [N, K] float32  (computes input @ weight.T)
// bias  : [N]    float32  (optional)
// returns [M, N] float32
// ---------------------------------------------------------------------------
torch::Tensor fp8_linear_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias
) {
    TORCH_CHECK(input.is_cuda()  && input.is_contiguous(),
                "fp8_linear_forward: input must be a contiguous CUDA tensor");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(),
                "fp8_linear_forward: weight must be a contiguous CUDA tensor");
    int input_dtype = float_dtype_code(input, "fp8_linear_forward: input");
    int weight_dtype = float_dtype_code(weight, "fp8_linear_forward: weight");
    TORCH_CHECK(input.dim()  == 2, "fp8_linear_forward: input must be 2-D");
    TORCH_CHECK(weight.dim() == 2, "fp8_linear_forward: weight must be 2-D");

    int64_t M = input.size(0);
    int64_t K = input.size(1);
    int64_t N = weight.size(0);
    TORCH_CHECK(weight.size(1) == K,
                "fp8_linear_forward: weight K-dim mismatch");

    // Bug 2: guard int64 → int narrowing
    int iM = checked_int(M, "M");
    int iN = checked_int(N, "N");
    int iK = checked_int(K, "K");

    // Bug 3: guard HW grid dimension limits
    check_grid_dim(M, 16, "M (rows)");
    check_grid_dim(N, 16, "N (cols)");

    // Hazard B: both tensors must be on the same device
    TORCH_CHECK(input.device() == weight.device(),
                "fp8_linear_forward: input and weight must be on the same device");
    check_gfx12_fp8_wmma_runtime("fp8_linear_forward");

    auto output = torch::zeros({M, N}, input.options());
    launch_fp8_linear_forward(
        input.data_ptr(),
        weight.data_ptr(),
        output.data_ptr(),
        iM, iN, iK, input_dtype, weight_dtype, input_dtype,
        current_stream()
    );

    if (bias.has_value()) {
        auto b = bias.value();
        TORCH_CHECK(b.is_cuda() && b.is_contiguous(),
                    "fp8_linear_forward: bias must be a contiguous CUDA tensor");
        TORCH_CHECK(b.size(0) == N,
                    "fp8_linear_forward: bias size mismatch");
        // Bug 4: validate bias dtype; mismatched dtype causes silent upcasting
        (void)float_dtype_code(b, "fp8_linear_forward: bias");
        TORCH_CHECK(b.device() == output.device(),
                    "fp8_linear_forward: bias must be on the same device as input");
        output.add_(b.unsqueeze(0));
    }

    return output;
}

torch::Tensor fp8_linear_forward_scaled(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    double input_scale,
    double weight_scale
) {
    TORCH_CHECK(input.is_cuda()  && input.is_contiguous(),
                "fp8_linear_forward_scaled: input must be a contiguous CUDA tensor");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(),
                "fp8_linear_forward_scaled: weight must be a contiguous CUDA tensor");
    int input_dtype = float_dtype_code(input, "fp8_linear_forward_scaled: input");
    int weight_dtype = float_dtype_code(weight, "fp8_linear_forward_scaled: weight");
    TORCH_CHECK(input.dim()  == 2, "fp8_linear_forward_scaled: input must be 2-D");
    TORCH_CHECK(weight.dim() == 2, "fp8_linear_forward_scaled: weight must be 2-D");
    TORCH_CHECK(input_scale > 0.0 && weight_scale > 0.0,
                "fp8_linear_forward_scaled: scales must be positive");

    int64_t M = input.size(0);
    int64_t K = input.size(1);
    int64_t N = weight.size(0);
    TORCH_CHECK(weight.size(1) == K,
                "fp8_linear_forward_scaled: weight K-dim mismatch");

    int iM = checked_int(M, "M");
    int iN = checked_int(N, "N");
    int iK = checked_int(K, "K");
    check_grid_dim(M, 16, "M (rows)");
    check_grid_dim(N, 16, "N (cols)");
    TORCH_CHECK(input.device() == weight.device(),
                "fp8_linear_forward_scaled: input and weight must be on the same device");
    check_gfx12_fp8_wmma_runtime("fp8_linear_forward_scaled");

    auto output = torch::zeros({M, N}, input.options());
    launch_fp8_linear_forward_scaled(
        input.data_ptr(), weight.data_ptr(), output.data_ptr(),
        iM, iN, iK, (float)input_scale, (float)weight_scale,
        input_dtype, weight_dtype, input_dtype, current_stream());

    if (bias.has_value()) {
        auto b = bias.value();
        TORCH_CHECK(b.is_cuda() && b.is_contiguous(),
                    "fp8_linear_forward_scaled: bias must be a contiguous CUDA tensor");
        (void)float_dtype_code(b, "fp8_linear_forward_scaled: bias");
        TORCH_CHECK(b.size(0) == N,
                    "fp8_linear_forward_scaled: bias size mismatch");
        TORCH_CHECK(b.device() == output.device(),
                    "fp8_linear_forward_scaled: bias must be on the same device as input");
        output.add_(b.unsqueeze(0));
    }
    return output;
}

torch::Tensor fp8_linear_forward_fp8_weight(
    torch::Tensor input,
    torch::Tensor weight_fp8,
    double weight_inv_scale,
    double input_scale,
    c10::optional<torch::Tensor> bias
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "fp8_linear_forward_fp8_weight: input must be a contiguous CUDA tensor");
    TORCH_CHECK(weight_fp8.is_cuda() && weight_fp8.is_contiguous(),
                "fp8_linear_forward_fp8_weight: weight_fp8 must be a contiguous CUDA tensor");
    int input_dtype = float_dtype_code(input, "fp8_linear_forward_fp8_weight: input");
    TORCH_CHECK(weight_fp8.scalar_type() == torch::kUInt8,
                "fp8_linear_forward_fp8_weight: weight_fp8 must be uint8");
    TORCH_CHECK(input.dim() == 2, "fp8_linear_forward_fp8_weight: input must be 2-D");
    TORCH_CHECK(weight_fp8.dim() == 2, "fp8_linear_forward_fp8_weight: weight_fp8 must be 2-D");
    TORCH_CHECK(input_scale > 0.0 && weight_inv_scale > 0.0,
                "fp8_linear_forward_fp8_weight: scales must be positive");

    int64_t M = input.size(0);
    int64_t K = input.size(1);
    int64_t N = weight_fp8.size(0);
    TORCH_CHECK(weight_fp8.size(1) == K,
                "fp8_linear_forward_fp8_weight: weight K-dim mismatch");

    int iM = checked_int(M, "M");
    int iN = checked_int(N, "N");
    int iK = checked_int(K, "K");
    check_grid_dim(M, 16, "M (rows)");
    check_grid_dim(N, 16, "N (cols)");
    TORCH_CHECK(input.device() == weight_fp8.device(),
                "fp8_linear_forward_fp8_weight: input and weight_fp8 must be on the same device");
    check_gfx12_fp8_wmma_runtime("fp8_linear_forward_fp8_weight");

    auto output = torch::zeros({M, N}, input.options());
    launch_fp8_linear_forward_fp8_weight(
        input.data_ptr(), weight_fp8.data_ptr<uint8_t>(), output.data_ptr(),
        iM, iN, iK, (float)input_scale, (float)weight_inv_scale,
        input_dtype, input_dtype, current_stream());

    if (bias.has_value()) {
        auto b = bias.value();
        TORCH_CHECK(b.is_cuda() && b.is_contiguous(),
                    "fp8_linear_forward_fp8_weight: bias must be a contiguous CUDA tensor");
        (void)float_dtype_code(b, "fp8_linear_forward_fp8_weight: bias");
        TORCH_CHECK(b.size(0) == N,
                    "fp8_linear_forward_fp8_weight: bias size mismatch");
        TORCH_CHECK(b.device() == output.device(),
                    "fp8_linear_forward_fp8_weight: bias must be on the same device as input");
        output.add_(b.unsqueeze(0));
    }
    return output;
}

// ---------------------------------------------------------------------------
// Phase 4 — fp8_linear_backward_input
// grad_output: [M, N] float32
// weight      : [N, K] float32
// returns grad_input: [M, K] float32
// ---------------------------------------------------------------------------
torch::Tensor fp8_linear_backward_input(
    torch::Tensor grad_output,
    torch::Tensor weight
) {
    TORCH_CHECK(grad_output.is_cuda() && grad_output.is_contiguous(),
                "fp8_linear_backward_input: grad_output must be contiguous CUDA");
    TORCH_CHECK(weight.is_cuda()      && weight.is_contiguous(),
                "fp8_linear_backward_input: weight must be contiguous CUDA");
    int grad_output_dtype = float_dtype_code(grad_output, "fp8_linear_backward_input: grad_output");
    int weight_dtype = float_dtype_code(weight, "fp8_linear_backward_input: weight");

    int64_t M = grad_output.size(0);
    int64_t N = grad_output.size(1);
    int64_t K = weight.size(1);
    TORCH_CHECK(weight.size(0) == N,
                "fp8_linear_backward_input: grad_output / weight N-dim mismatch");

    // Bug 2: guard int64 → int narrowing
    int iM = checked_int(M, "M");
    int iN = checked_int(N, "N");
    int iK = checked_int(K, "K");

    // Bug 3: guard HW grid dimension limits
    check_grid_dim(M, 16, "M (rows)");
    check_grid_dim(K, 16, "K (cols)");

    // Hazard B: grad_output and weight must be on the same device
    TORCH_CHECK(grad_output.device() == weight.device(),
                "fp8_linear_backward_input: grad_output and weight must be on the same device");
    check_gfx12_fp8_wmma_runtime("fp8_linear_backward_input");

    auto grad_input = torch::zeros({M, K}, grad_output.options());
    launch_fp8_linear_backward_input(
        grad_output.data_ptr(),
        weight.data_ptr(),
        grad_input.data_ptr(),
        iM, iN, iK, grad_output_dtype, weight_dtype, grad_output_dtype,
        current_stream()
    );
    return grad_input;
}

torch::Tensor fp8_linear_backward_input_scaled(
    torch::Tensor grad_output,
    torch::Tensor weight,
    double weight_scale
) {
    TORCH_CHECK(grad_output.is_cuda() && grad_output.is_contiguous(),
                "fp8_linear_backward_input_scaled: grad_output must be contiguous CUDA");
    TORCH_CHECK(weight.is_cuda()      && weight.is_contiguous(),
                "fp8_linear_backward_input_scaled: weight must be contiguous CUDA");
    int grad_output_dtype = float_dtype_code(grad_output, "fp8_linear_backward_input_scaled: grad_output");
    int weight_dtype = float_dtype_code(weight, "fp8_linear_backward_input_scaled: weight");
    TORCH_CHECK(weight_scale > 0.0,
                "fp8_linear_backward_input_scaled: weight_scale must be positive");

    int64_t M = grad_output.size(0);
    int64_t N = grad_output.size(1);
    int64_t K = weight.size(1);
    TORCH_CHECK(weight.size(0) == N,
                "fp8_linear_backward_input_scaled: grad_output / weight N-dim mismatch");

    int iM = checked_int(M, "M");
    int iN = checked_int(N, "N");
    int iK = checked_int(K, "K");
    check_grid_dim(M, 16, "M (rows)");
    check_grid_dim(K, 16, "K (cols)");
    TORCH_CHECK(grad_output.device() == weight.device(),
                "fp8_linear_backward_input_scaled: grad_output and weight must be on the same device");
    check_gfx12_fp8_wmma_runtime("fp8_linear_backward_input_scaled");

    auto grad_input = torch::zeros({M, K}, grad_output.options());
    launch_fp8_linear_backward_input_scaled(
        grad_output.data_ptr(), weight.data_ptr(), grad_input.data_ptr(),
        iM, iN, iK, (float)weight_scale, grad_output_dtype, weight_dtype,
        grad_output_dtype, current_stream());
    return grad_input;
}

// ---------------------------------------------------------------------------
// Phase 4 — fp8_linear_backward_weight
// grad_output: [M, N] float32
// input       : [M, K] float32
// returns grad_weight: [N, K] float32
// ---------------------------------------------------------------------------
torch::Tensor fp8_linear_backward_weight(
    torch::Tensor grad_output,
    torch::Tensor input
) {
    TORCH_CHECK(grad_output.is_cuda() && grad_output.is_contiguous(),
                "fp8_linear_backward_weight: grad_output must be contiguous CUDA");
    TORCH_CHECK(input.is_cuda()       && input.is_contiguous(),
                "fp8_linear_backward_weight: input must be contiguous CUDA");
    int grad_output_dtype = float_dtype_code(grad_output, "fp8_linear_backward_weight: grad_output");
    int input_dtype = float_dtype_code(input, "fp8_linear_backward_weight: input");

    int64_t M = grad_output.size(0);
    int64_t N = grad_output.size(1);
    int64_t K = input.size(1);
    TORCH_CHECK(input.size(0) == M,
                "fp8_linear_backward_weight: grad_output / input M-dim mismatch");

    // Bug 2: guard int64 → int narrowing
    int iM = checked_int(M, "M");
    int iN = checked_int(N, "N");
    int iK = checked_int(K, "K");

    // Bug 3: guard HW grid dimension limits
    check_grid_dim(N, 16, "N (rows)");
    check_grid_dim(K, 16, "K (cols)");

    // Hazard B: grad_output and input must be on the same device
    TORCH_CHECK(grad_output.device() == input.device(),
                "fp8_linear_backward_weight: grad_output and input must be on the same device");
    check_gfx12_fp8_wmma_runtime("fp8_linear_backward_weight");

    auto grad_weight = torch::zeros({N, K}, grad_output.options());
    launch_fp8_linear_backward_weight(
        grad_output.data_ptr(),
        input.data_ptr(),
        grad_weight.data_ptr(),
        iM, iN, iK, grad_output_dtype, input_dtype, grad_output_dtype,
        current_stream()
    );
    return grad_weight;
}

torch::Tensor fp8_linear_backward_weight_scaled(
    torch::Tensor grad_output,
    torch::Tensor input,
    double input_scale
) {
    TORCH_CHECK(grad_output.is_cuda() && grad_output.is_contiguous(),
                "fp8_linear_backward_weight_scaled: grad_output must be contiguous CUDA");
    TORCH_CHECK(input.is_cuda()       && input.is_contiguous(),
                "fp8_linear_backward_weight_scaled: input must be contiguous CUDA");
    int grad_output_dtype = float_dtype_code(grad_output, "fp8_linear_backward_weight_scaled: grad_output");
    int input_dtype = float_dtype_code(input, "fp8_linear_backward_weight_scaled: input");
    TORCH_CHECK(input_scale > 0.0,
                "fp8_linear_backward_weight_scaled: input_scale must be positive");

    int64_t M = grad_output.size(0);
    int64_t N = grad_output.size(1);
    int64_t K = input.size(1);
    TORCH_CHECK(input.size(0) == M,
                "fp8_linear_backward_weight_scaled: grad_output / input M-dim mismatch");

    int iM = checked_int(M, "M");
    int iN = checked_int(N, "N");
    int iK = checked_int(K, "K");
    check_grid_dim(N, 16, "N (rows)");
    check_grid_dim(K, 16, "K (cols)");
    TORCH_CHECK(grad_output.device() == input.device(),
                "fp8_linear_backward_weight_scaled: grad_output and input must be on the same device");
    check_gfx12_fp8_wmma_runtime("fp8_linear_backward_weight_scaled");

    auto grad_weight = torch::zeros({N, K}, grad_output.options());
    launch_fp8_linear_backward_weight_scaled(
        grad_output.data_ptr(), input.data_ptr(), grad_weight.data_ptr(),
        iM, iN, iK, (float)input_scale, grad_output_dtype, input_dtype,
        grad_output_dtype, current_stream());
    return grad_weight;
}

// ---------------------------------------------------------------------------
// Pybind11 module registration
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "hip_quant PyTorch FP8 extension — AMD ROCm / HIP";

    // Phase 1
    m.def("quantize_e4m3",   &quantize_e4m3,
          "Quantize float32 tensor to FP8 E4M3 (uint8) on-device",
          py::arg("input"));
    m.def("quantize_e5m2",   &quantize_e5m2,
          "Quantize float32 tensor to FP8 E5M2 (uint8) on-device",
          py::arg("input"));
    m.def("dequantize_e4m3", &dequantize_e4m3,
          "Dequantize FP8 E4M3 (uint8) tensor to float32 on-device",
          py::arg("input"));
    m.def("dequantize_e5m2", &dequantize_e5m2,
          "Dequantize FP8 E5M2 (uint8) tensor to float32 on-device",
          py::arg("input"));

    // Phase 4
    m.def("fp8_linear_forward",         &fp8_linear_forward,
          "FP8 linear forward: output = quant(input) @ quant(weight).T + bias",
          py::arg("input"), py::arg("weight"),
          py::arg("bias") = c10::optional<torch::Tensor>());
    m.def("fp8_linear_forward_scaled",  &fp8_linear_forward_scaled,
          "Scaled FP8 linear forward using E4M3 WMMA",
          py::arg("input"), py::arg("weight"),
          py::arg("bias") = c10::optional<torch::Tensor>(),
          py::arg("input_scale"), py::arg("weight_scale"));
    m.def("fp8_linear_forward_fp8_weight", &fp8_linear_forward_fp8_weight,
          "Scaled FP8 linear forward using a pre-quantized E4M3 weight buffer",
          py::arg("input"), py::arg("weight_fp8"),
          py::arg("weight_inv_scale"), py::arg("input_scale"),
          py::arg("bias") = c10::optional<torch::Tensor>());
    m.def("fp8_linear_backward_input",  &fp8_linear_backward_input,
          "FP8 linear backward — grad w.r.t. input",
          py::arg("grad_output"), py::arg("weight"));
    m.def("fp8_linear_backward_input_scaled", &fp8_linear_backward_input_scaled,
          "Scaled FP8 linear backward — grad w.r.t. input",
          py::arg("grad_output"), py::arg("weight"), py::arg("weight_scale"));
    m.def("fp8_linear_backward_weight", &fp8_linear_backward_weight,
          "FP8 linear backward — grad w.r.t. weight",
          py::arg("grad_output"), py::arg("input"));
    m.def("fp8_linear_backward_weight_scaled", &fp8_linear_backward_weight_scaled,
          "Scaled FP8 linear backward — grad w.r.t. weight",
          py::arg("grad_output"), py::arg("input"), py::arg("input_scale"));
}
