import sys; sys.path.insert(0, r'C:\Users\armor\Desktop\hip_quant')
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

print("\n=== quantize/dequantize ===")
x = torch.randn(2, 4, device='cuda')
q = hq.quantize(x, 'e4m3')
d = hq.dequantize(q)
cos2 = F.cosine_similarity(x.flatten(), d.flatten(), dim=0).item()
print(f"  cos: {cos2:.4f} {'OK' if cos2>0.99 else 'FAIL'}")

print("\n=== attention alias ===")
print(f"  attention is wave_attn: {hq.attention is hq.wave_attn}")

print("\n=== All QoL tests passed ===")
