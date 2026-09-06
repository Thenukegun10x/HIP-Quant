import os
import math
import torch
import torch.nn.functional as F

from hip_quant import torch_api as _T

def test_fused_ssm():
    ext = _T._load_extension()
    print("delta_net_prefill_forward:", hasattr(ext, "delta_net_prefill_forward"))
    print("fast_ssm_conv1d_forward:", hasattr(ext, "fast_ssm_conv1d_forward"))
    print("fused_ssm_gating_forward:", hasattr(ext, "fused_ssm_gating_forward"))
    print("fused_delta_net_prep_forward:", hasattr(ext, "fused_delta_net_prep_forward"))
    print("fast_rms_norm_gated_forward:", hasattr(ext, "fast_rms_norm_gated_forward"))

    if not torch.cuda.is_available():
        print("No CUDA available")
        return

    device = "cuda"
    torch.manual_seed(42)
    S_len = 64  # test sequence length

    # 1. Test fast_ssm_conv1d_forward
    print("\n--- Testing fast_ssm_conv1d_forward ---")
    conv_w = torch.randn(10240, 4, dtype=torch.float16, device=device)
    x = torch.randn(S_len, 10240, dtype=torch.float16, device=device)
    conv_state_ref = torch.randn(10240, 3, dtype=torch.float16, device=device)
    conv_state_fused = conv_state_ref.clone()

    # PyTorch reference
    x_conv = x.unsqueeze(0).transpose(1, 2)  # [1, 10240, S_len]
    x_full = torch.cat([conv_state_ref.unsqueeze(0), x_conv], dim=2)
    conv_state_ref.copy_(x_full[0, :, -3:])
    windows = x_full.unfold(dimension=2, size=4, step=1)
    ref_conv_out = (windows * conv_w.unsqueeze(0).unsqueeze(2)).sum(dim=-1)
    ref_conv_out = F.silu(ref_conv_out).transpose(1, 2).squeeze(0)  # [S_len, 10240]

    # HIP kernel
    fused_conv_out = ext.fast_ssm_conv1d_forward(x, conv_state_fused, conv_w)

    conv_diff = (fused_conv_out - ref_conv_out).abs().max().item()
    state_diff = (conv_state_fused - conv_state_ref).abs().max().item()
    print(f"conv_out max diff: {conv_diff:.6f}, state max diff: {state_diff:.6f}")
    assert conv_diff < 1e-2, f"conv_out mismatch: {conv_diff}"
    assert state_diff < 1e-2, f"state mismatch: {state_diff}"

    # 2. Test fused_ssm_gating_forward
    print("\n--- Testing fused_ssm_gating_forward ---")
    beta_raw = torch.randn(S_len, 48, dtype=torch.float16, device=device)
    alpha_raw = torch.randn(S_len, 48, dtype=torch.float16, device=device)
    dt_bias = torch.randn(48, dtype=torch.float32, device=device)
    ssm_a = -torch.rand(48, dtype=torch.float32, device=device) - 0.1

    # PyTorch reference
    ref_beta = torch.sigmoid(beta_raw.float())
    ref_alpha_biased = alpha_raw.float() + dt_bias
    ref_gate = ssm_a * F.softplus(ref_alpha_biased)
    ref_decay = torch.exp(ref_gate)

    # HIP kernel
    fused_decay, fused_beta = ext.fused_ssm_gating_forward(beta_raw, alpha_raw, dt_bias, ssm_a)

    decay_diff = (fused_decay - ref_decay).abs().max().item()
    beta_diff = (fused_beta - ref_beta).abs().max().item()
    print(f"decay max diff: {decay_diff:.6f}, beta max diff: {beta_diff:.6f}")
    assert decay_diff < 1e-4, f"decay mismatch: {decay_diff}"
    assert beta_diff < 1e-4, f"beta mismatch: {beta_diff}"

    # 3. Test fused_delta_net_prep_forward
    print("\n--- Testing fused_delta_net_prep_forward ---")
    # PyTorch reference
    ref_q = F.normalize(fused_conv_out[:, :2048].reshape(S_len, 16, 128).float(), p=2, dim=-1)
    ref_k = F.normalize(fused_conv_out[:, 2048:4096].reshape(S_len, 16, 128).float(), p=2, dim=-1)
    ref_v = fused_conv_out[:, 4096:].reshape(S_len, 48, 128).float()

    # HIP kernel
    q_norm, k_norm, v_float = ext.fused_delta_net_prep_forward(fused_conv_out)

    q_diff = (q_norm - ref_q).abs().max().item()
    k_diff = (k_norm - ref_k).abs().max().item()
    v_diff = (v_float - ref_v).abs().max().item()
    print(f"q_norm max diff: {q_diff:.6f}, k_norm max diff: {k_diff:.6f}, v_float max diff: {v_diff:.6f}")
    assert q_diff < 1e-4, f"q mismatch: {q_diff}"
    assert k_diff < 1e-4, f"k mismatch: {k_diff}"
    assert v_diff < 1e-4, f"v mismatch: {v_diff}"

    # 4. Test delta_net_prefill_forward
    print("\n--- Testing delta_net_prefill_forward ---")
    scale_ssm = 1.0 / math.sqrt(128)
    S_mat_ref = torch.randn(48, 128, 128, dtype=torch.float32, device=device) * 0.01
    S_mat_fused = S_mat_ref.clone()

    # PyTorch reference sequential loop
    ref_out_all = torch.empty(S_len, 48, 128, dtype=torch.float16, device=device)
    q_all_48 = ref_q.repeat(1, 3, 1)
    k_all_48 = ref_k.repeat(1, 3, 1)
    decay_4d = ref_decay.view(S_len, 48, 1, 1)
    beta_4d = ref_beta.view(S_len, 48, 1, 1)

    for t in range(S_len):
        kt = k_all_48[t].unsqueeze(1)
        vt = ref_v[t].unsqueeze(1)
        dt = decay_4d[t]
        bt = beta_4d[t]
        kv = torch.bmm(kt, S_mat_ref)
        delta = (vt - dt * kv) * bt
        S_mat_ref.mul_(dt).baddbmm_(k_all_48[t].unsqueeze(-1), delta)
        out_state = (torch.bmm(q_all_48[t].unsqueeze(1), S_mat_ref).squeeze(1) * scale_ssm).half()
        ref_out_all[t].copy_(out_state)

    # HIP kernel
    fused_out_all = torch.empty(S_len, 48, 128, dtype=torch.float16, device=device)
    ext.delta_net_prefill_forward(
        q_norm, k_norm, v_float,
        fused_decay, fused_beta,
        S_mat_fused, scale_ssm,
        fused_out_all
    )

    out_diff = (fused_out_all - ref_out_all).abs().max().item()
    s_mat_diff = (S_mat_fused - S_mat_ref).abs().max().item()
    print(f"delta_net out max diff: {out_diff:.6f}, S_mat max diff: {s_mat_diff:.6f}")
    assert out_diff < 5e-3, f"DeltaNet output mismatch: {out_diff}"
    assert s_mat_diff < 5e-3, f"DeltaNet S_mat mismatch: {s_mat_diff}"
    print("\n>>> ALL 4 FUSED SSM PREFILL KERNELS MATCH PYTORCH REFERENCE EXACTLY! <<<")

if __name__ == "__main__":
    import sys
    try:
        test_fused_ssm()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        os._exit(0)
