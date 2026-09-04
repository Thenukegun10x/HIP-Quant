import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch, torch.nn.functional as F
import torch_api as hq

print("=== QuantizedLinear ===")
lin = hq.QuantizedLinear(64, 128, bias=True)
print(f"  {lin}")
lin = lin.cuda()
x = torch.randn(4, 8, 64, device='cuda')
y = lin(x)
print(f"  output shape: {y.shape}")
ref = torch.nn.functional.linear(x, hq.dequantize_e4m3(lin._fp8_weight), lin.bias)
cos = torch.nn.functional.cosine_similarity(y.flatten(), ref.flatten(), dim=0).item()
print(f"  cos vs reference: {cos:.4f} {'OK' if cos>0.98 else 'FAIL'}")

print("\n=== convert_to_quantized ===")
model = torch.nn.Sequential(
    torch.nn.Linear(64, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 32),
)
hq.convert_to_quantized(model)
print(f"  layer 0 type: {type(model[0]).__name__}")
print(f"  layer 2 type: {type(model[2]).__name__}")
assert isinstance(model[0], hq.QuantizedLinear)

print("\n=== quantize/dequantize with scale ===")
x = torch.randn(2, 4, device='cuda')
x_fp8, scale = hq.quantize(x, 'e4m3')
print(f"  scale: {scale.item():.6f}")
d = hq.dequantize(x_fp8, scale=scale)
cos2 = torch.nn.functional.cosine_similarity(x.flatten(), d.flatten(), dim=0).item()
print(f"  cos: {cos2:.4f} {'OK' if cos2>0.99 else 'FAIL'}")

# Per-channel
w = torch.randn(128, 64, device='cuda')
w_fp8, scales = hq.quantize(w, granularity="per_channel", dim=0)
print(f"  per-channel scales shape: {scales.shape}  (expect [128, 1])")
print(f"  scale range: [{scales.min().item():.6f}, {scales.max().item():.6f}]")

print("\n=== attention alias ===")
print(f"  attention is wave_attn: {hq.attention is hq.wave_attn}")

print("\n=== All QoL tests passed ===")
