// Current PyTorch HIP stream bridge.
//
// This file is compiled by hipcc/clang rather than MSVC.  The Windows ROCm
// PyTorch wheel currently exposes c10 HIP stream APIs whose header-only helper
// code uses clang-specific attributes, so including that header in the MSVC
// binding translation unit fails.  Keeping the bridge here preserves PyTorch
// stream semantics without making the host binding depend on those headers.

#include <hip/hip_runtime.h>
#include <c10/hip/HIPStream.h>

extern "C" hipStream_t hip_quant_get_current_stream() {
    return c10::hip::getCurrentHIPStream().stream();
}
