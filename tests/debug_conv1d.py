import os
import sys
import torch
from hip_quant import torch_api as _T

def main():
    ext = _T._load_extension()
    device = "cuda"
    torch.manual_seed(42)

    conv_w = torch.randn(10240, 4, dtype=torch.float16, device=device)
    cs = torch.randn(10240, 3, dtype=torch.float16, device=device)

    for s in [1, 2, 3, 4, 8, 16, 32, 64]:
        x = torch.randn(s, 10240, dtype=torch.float16, device=device)
        c_cur = cs.clone()
        out = ext.fast_ssm_conv1d_forward(x, c_cur, conv_w)
        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        print(f"s={s:2d}: isnan={has_nan}, isinf={has_inf}, min={out.min().item():.2f}, max={out.max().item():.2f}")
        if has_nan:
            nan_pos = torch.nonzero(torch.isnan(out))[0].tolist()
            t_idx, c_idx = nan_pos[0], nan_pos[1]
            print(f"   first nan at token {t_idx}, channel {c_idx}")
            print(f"   x for channel {c_idx}: {x[:, c_idx].tolist()}")
            print(f"   out for channel {c_idx}: {out[:, c_idx].tolist()}")
            print(f"   conv_w for channel {c_idx}: {conv_w[c_idx].tolist()}")
            print(f"   cs for channel {c_idx}: {cs[c_idx].tolist()}")
            break

if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        os._exit(0)
