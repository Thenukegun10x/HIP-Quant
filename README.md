<div align="center">
  <h1>🚀 hip-quant</h1>
  <p><b>Blazing Fast On-Device Tensor Quantization for AMD GPUs</b></p>
</div>

`hip-quant` is a standalone Python library and highly optimized HIP C++ backend designed to instantaneously quantize standard `float32` tensors directly on AMD GPUs. It bypasses CPU bottlenecks by performing parallel warp reductions and bit-packing entirely on the device (targeting ROCm 7.1 and the `gfx1201` architecture).

Whether you are converting massive LLMs on the fly for streaming layers to VRAM, or performing dynamic per-layer sensitivity analysis, `hip-quant` produces 100% byte-for-byte exact matches to the `llama.cpp` CPU reference implementations at a fraction of the time.

---

## ⚡ Supported Formats

`hip-quant` supports virtually every modern GGUF quantization format, making it incredibly versatile for large language model inference.

### 🔢 Standard Integer & K-Quants
- **Legacy:** `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`
- **K-Quants:** `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`

### 🧠 I-Quants (Importance Matrix)
Optimized non-linear quants that preserve quality at extreme low bits:
- `IQ1_S`, `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`

### ⚖️ Ternary Quants (Base-3)
*Note: Ternary quants map weights strictly to three states (-1, 0, 1) and pack them dynamically in Base-3. These are incredibly memory and compute efficient but are intended exclusively for models trained to be ternary (e.g., BitNet, TriLM).*
- `TQ1_0` (1.69 bpw)
- `TQ2_0` (2.06 bpw)

---

## 🛠️ Build Instructions

To compile the C++ source into the required Windows DLL (`hip_quantize.dll`), simply invoke the PowerShell script. 

*Note: This strictly requires `hipcc` located at `C:\Program Files\AMD\ROCm\7.1\bin\hipcc.exe`.*

```powershell
.\build.ps1
```

## 📦 Installation & Usage

You can build and install the Python wrapper directly via standard Python tools:
```powershell
python -m build
pip install dist/hip_quant-0.1.0-py3-none-any.whl
```

### Python API Example
```python
import numpy as np
from hip_quant import quantize

# Create a dummy float32 weight matrix
hidden_size = 4096
weights = np.random.randn(hidden_size, hidden_size).astype(np.float32)

# Instantly quantize directly to Q4_K on the GPU!
# Returns a tightly packed uint8 byte array exactly matching GGUF formats
q4k_bytes = quantize(weights, type_num=12) # 12 = Q4_K
```

---

## 🤖 Dynamic Agentic Profiles (Integration)

`hip-quant` seamlessly integrates with dynamic quantization pipelines. For example, you can safely slice 8GB weight matrices into manageable rows and batch them on the GPU to prevent VRAM OOMs:

```python
import numpy as np
from hip_quant import get_hip_quant

hq = get_hip_quant()
target_type = 23 # IQ4_XS
nrows, n_per_row = 4096, 4096
bytes_per_row = hq.row_size(target_type, n_per_row)
dst = np.empty(nrows * bytes_per_row, dtype=np.uint8)

# Safe VRAM Batching
batch_size = 4096
for start_row in range(0, nrows, batch_size):
    end_row = min(nrows, start_row + batch_size)
    batch = f32_data[start_row:end_row]
    dst_slice = dst[start_row * bytes_per_row : end_row * bytes_per_row]
    hq.quantize_numpy_to(batch, target_type, dst_slice)
```
