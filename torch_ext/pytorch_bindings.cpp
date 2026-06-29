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
#include <pybind11/pybind11.h>
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

typedef struct ihipStream_t* hipStream_t;

// Forward declarations for kernel launchers defined in the .hip files.
extern "C" {
void launch_quant_e4m3(const float* src, uint8_t* dst, int64_t numel,
                       hipStream_t stream);
void launch_quant_e5m2(const float* src, uint8_t* dst, int64_t numel,
                       hipStream_t stream);
void launch_dequant_e4m3(const uint8_t* src, float* dst, int64_t numel,
                         hipStream_t stream);
void launch_dequant_e5m2(const uint8_t* src, float* dst, int64_t numel,
                         hipStream_t stream);

void launch_fp8_linear_forward(
    const float* A, const float* B, float* C,
    int M, int N, int K, hipStream_t stream);
void launch_fp8_linear_backward_input(
    const float* grad_output, const float* weight, float* grad_input,
    int M, int N, int K, hipStream_t stream);
void launch_fp8_linear_backward_weight(
    const float* grad_output, const float* input, float* grad_weight,
    int M, int N, int K, hipStream_t stream);
} // extern "C"

// ---------------------------------------------------------------------------
// Helper: get current HIP stream from ATen
// ---------------------------------------------------------------------------
static hipStream_t current_stream() {
    return nullptr;
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

// ---------------------------------------------------------------------------
// Phase 1 — quantize_e4m3
// ---------------------------------------------------------------------------
torch::Tensor quantize_e4m3(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(),       "quantize_e4m3: input must be a HIP/CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "quantize_e4m3: input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "quantize_e4m3: input must be float32");

    auto output = torch::empty(input.sizes(),
                               input.options().dtype(torch::kUInt8));
    int64_t numel = input.numel();
    launch_quant_e4m3(
        input.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
        numel,
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
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "quantize_e5m2: input must be float32");

    auto output = torch::empty(input.sizes(),
                               input.options().dtype(torch::kUInt8));
    int64_t numel = input.numel();
    launch_quant_e5m2(
        input.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
        numel,
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
    TORCH_CHECK(input.scalar_type()  == torch::kFloat32,
                "fp8_linear_forward: input must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32,
                "fp8_linear_forward: weight must be float32");
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

    auto output = torch::zeros({M, N}, input.options());
    launch_fp8_linear_forward(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        iM, iN, iK,
        current_stream()
    );

    if (bias.has_value()) {
        auto b = bias.value();
        TORCH_CHECK(b.is_cuda() && b.is_contiguous(),
                    "fp8_linear_forward: bias must be a contiguous CUDA tensor");
        TORCH_CHECK(b.size(0) == N,
                    "fp8_linear_forward: bias size mismatch");
        // Bug 4: validate bias dtype; mismatched dtype causes silent upcasting
        TORCH_CHECK(b.scalar_type() == torch::kFloat32,
                    "fp8_linear_forward: bias must be float32");
        TORCH_CHECK(b.device() == output.device(),
                    "fp8_linear_forward: bias must be on the same device as input");
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
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32,
                "fp8_linear_backward_input: grad_output must be float32");
    TORCH_CHECK(weight.scalar_type()      == torch::kFloat32,
                "fp8_linear_backward_input: weight must be float32");

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

    auto grad_input = torch::zeros({M, K}, grad_output.options());
    launch_fp8_linear_backward_input(
        grad_output.data_ptr<float>(),
        weight.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        iM, iN, iK,
        current_stream()
    );
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
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32,
                "fp8_linear_backward_weight: grad_output must be float32");
    TORCH_CHECK(input.scalar_type()       == torch::kFloat32,
                "fp8_linear_backward_weight: input must be float32");

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

    auto grad_weight = torch::zeros({N, K}, grad_output.options());
    launch_fp8_linear_backward_weight(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        grad_weight.data_ptr<float>(),
        iM, iN, iK,
        current_stream()
    );
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
    m.def("fp8_linear_backward_input",  &fp8_linear_backward_input,
          "FP8 linear backward — grad w.r.t. input",
          py::arg("grad_output"), py::arg("weight"));
    m.def("fp8_linear_backward_weight", &fp8_linear_backward_weight,
          "FP8 linear backward — grad w.r.t. weight",
          py::arg("grad_output"), py::arg("input"));
}
