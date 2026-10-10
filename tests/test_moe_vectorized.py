"""T012 (FR6, Plan B) tests: MoE vectorized rewrite of the SwiGLU expert path.

Red-phase semantics
-------------------
The current implementation in ``model/modern_gpt.py`` IS the old (loop-based)
implementation. Therefore:

* "pin" tests (3-10) must PASS right now: they lock the observable behavior
  (forward equivalence, int() capacity truncation, dropped-token sets, legacy
  state_dict key format, dense path for num_experts<=1, aux loss) so the
  rewrite cannot change it.
* structure / static-source / benchmark tests (1, 2, 11) must FAIL right now:
  they demand the NEW structure (stacked Parameters ``gate_w/up_w/down_w``,
  no per-expert Python loop / torch.stack / numel() in forward, >= 2x speedup).
* test 12 (compile stability) is expected to SKIP on Windows where
  torch.compile with the inductor backend is unavailable.

The old behavior is preserved in this file as ``_reference_moe_forward``, a
faithful replica of the pre-T012 MoE forward (torch.stack + per-expert Python
loop + per-expert capacity truncation). It is the ground truth for all
equivalence and dropped-set assertions and the baseline for the speedup test.

All tests run on CPU (torch 2.5.1). Weights are always Linear-init scale
(uniform +-1/sqrt(fan_in)); randn weights would blow up activation magnitudes
and make atol=1e-5 comparisons unsafe.
"""

import inspect
import math
import statistics
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ModernGPT, ModernGPTConfig
from model.modern_gpt import SwiGLU


# ---------------------------------------------------------------------------
# Reference (old) implementation: faithful replica of the pre-T012 MoE forward
# ---------------------------------------------------------------------------

def _reference_moe_forward(x, gate_list, up_list, down_list, router_weight,
                           capacity_factor, aux_loss_factor):
    """Faithful replica of the current (pre-T012) MoE forward.

    Parameters
    ----------
    x : Tensor [B, T, C]
    gate_list, up_list : list of E tensors, each [H, C]
    down_list : list of E tensors, each [C, H]
    router_weight : Tensor [E, C]
    capacity_factor, aux_loss_factor : float

    Returns (out [B, T, C], aux_loss scalar). No dropout is applied: every
    comparison in this file uses dropout=0.0 / eval mode, where the module's
    dropout is the identity.
    """
    B, T, C = x.shape
    E = len(gate_list)
    x_flat = x.reshape(B * T, C)  # [N, C]

    # routing scores
    router_logits = F.linear(x_flat, router_weight)  # [N, E]
    router_probs = F.softmax(router_logits, dim=-1)

    # top-1 selection
    _, selected = torch.topk(router_probs, 1, dim=-1)  # [N, 1]
    selected = selected.squeeze(-1)  # [N]

    # load-balancing aux loss (vectorized, identical to the module)
    token_counts = torch.bincount(selected, minlength=E).float()
    f = token_counts / (B * T)
    P = router_probs.mean(dim=0)
    aux_loss = aux_loss_factor * E * (f * P).sum()

    # per-expert capacity, int() truncation (Switch Transformer style)
    capacity = int(capacity_factor * B * T / E)

    # stack expert weights, then dispatch via a Python per-expert loop
    gate_w = torch.stack(gate_list, dim=0)  # [E, H, C]
    up_w = torch.stack(up_list, dim=0)  # [E, H, C]
    down_w = torch.stack(down_list, dim=0)  # [E, C, H]

    out_flat = torch.zeros_like(x_flat)
    for e in range(E):
        idx = (selected == e).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        if idx.numel() > capacity:
            idx = idx[:capacity]  # keep first `capacity` tokens in token order
        xe = x_flat.index_select(0, idx)  # [n_e, C]
        xe_3d = xe.unsqueeze(1)  # [n_e, 1, C]
        gw = gate_w[e].transpose(0, 1).unsqueeze(0).expand(xe.size(0), -1, -1)
        uw = up_w[e].transpose(0, 1).unsqueeze(0).expand(xe.size(0), -1, -1)
        dw = down_w[e].transpose(0, 1).unsqueeze(0).expand(xe.size(0), -1, -1)
        ge = F.silu(torch.bmm(xe_3d, gw).squeeze(1))  # [n_e, H]
        ue = torch.bmm(xe_3d, uw).squeeze(1)  # [n_e, H]
        de = torch.bmm((ge * ue).unsqueeze(1), dw).squeeze(1)  # [n_e, C]
        weight = router_probs.index_select(0, idx)[:, e].unsqueeze(-1)
        out_flat.index_copy_(0, idx, de * weight)

    out = out_flat.view(B, T, C)
    return out, aux_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_moe_cfg(num_experts, **overrides):
    kwargs = dict(
        n_layer=1,
        n_head=2,
        n_embd=32,
        block_size=16,
        num_experts=num_experts,
        dropout=0.0,
    )
    kwargs.update(overrides)
    return ModernGPTConfig(**kwargs)


def _make_tiny_model(num_experts, seed, **overrides):
    torch.manual_seed(seed)
    cfg = _make_moe_cfg(num_experts, **overrides)
    model = ModernGPT(cfg)
    model.eval()
    return cfg, model


def _extract_moe_weights(mlp):
    """Extract MoE weights via the legacy state_dict key format.

    Works both for the old (ModuleList) and the new (stacked Parameter)
    implementation because the on-disk key format is part of the contract.
    """
    sd = mlp.state_dict()
    E = mlp.num_experts
    gate_list = [sd[f"gate_proj.{e}.weight"] for e in range(E)]
    up_list = [sd[f"up_proj.{e}.weight"] for e in range(E)]
    down_list = [sd[f"down_proj.{e}.weight"] for e in range(E)]
    router_w = sd["router.weight"]
    return gate_list, up_list, down_list, router_w


def _uniform_linear_weight(shape, fan_in, seed):
    """Linear-init scale weight: uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    torch.manual_seed(seed)
    bound = 1.0 / math.sqrt(fan_in)
    return torch.empty(*shape).uniform_(-bound, bound).requires_grad_(True)


def _median_ms(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


# ---------------------------------------------------------------------------
# 1. Structure (Red: requires the new stacked-parameter layout)
# ---------------------------------------------------------------------------

def test_param_structure_stacked():
    """MoE branch must register stacked Parameters gate_w/up_w/down_w.

    Target layout: gate_w [E, H, C], up_w [E, H, C], down_w [E, C, H],
    registered directly as nn.Parameter (no ModuleList, no per-forward stack).
    """
    cfg, model = _make_tiny_model(num_experts=4, seed=1111)
    mlp = model.transformer.h[0].mlp
    E, H, C = cfg.num_experts, cfg.intermediate_size, cfg.n_embd

    for name, shape in (
        ("gate_w", (E, H, C)),
        ("up_w", (E, H, C)),
        ("down_w", (E, C, H)),
    ):
        assert hasattr(mlp, name), (
            f"MoE module must expose stacked parameter '{name}' "
            f"(got attributes: {sorted(n for n, _ in mlp.named_parameters())})"
        )
        p = getattr(mlp, name)
        assert isinstance(p, nn.Parameter), (
            f"mlp.{name} must be an nn.Parameter, got {type(p).__name__}"
        )
        assert tuple(p.shape) == shape, (
            f"mlp.{name} must have shape {shape}, got {tuple(p.shape)}"
        )

    assert not (hasattr(mlp, "gate_proj") and isinstance(mlp.gate_proj, nn.ModuleList)), (
        "mlp.gate_proj must no longer be an nn.ModuleList in the MoE branch"
    )


# ---------------------------------------------------------------------------
# 2. Static source contract (Red: forward must be loop-free / stack-free)
# ---------------------------------------------------------------------------

def test_forward_no_python_loop():
    """SwiGLU.forward must not contain a per-expert Python loop.

    Static assertions on the source: no 'for e in', no 'torch.stack', no
    'numel()' (data-dependent branching). The class as a whole must register
    a load_state_dict pre-hook for legacy checkpoint compatibility.
    """
    src_fwd = inspect.getsource(SwiGLU.forward)
    assert "for e in" not in src_fwd, (
        "SwiGLU.forward must not contain a per-expert Python loop ('for e in')"
    )
    assert "torch.stack" not in src_fwd, (
        "SwiGLU.forward must not call torch.stack (weights must be pre-stacked "
        "parameters)"
    )
    assert "numel()" not in src_fwd, (
        "SwiGLU.forward must not contain numel()-based data-dependent branches"
    )

    src_cls = inspect.getsource(SwiGLU)
    assert "_register_load_state_dict_pre_hook" in src_cls, (
        "SwiGLU must register a load_state_dict pre-hook "
        "(_register_load_state_dict_pre_hook) for legacy checkpoint compat"
    )


# ---------------------------------------------------------------------------
# 3. Forward equivalence vs reference (pin)
# ---------------------------------------------------------------------------

def test_forward_equivalence_vs_reference():
    """Model MoE forward must match the loop-based reference bit-for-bit.

    Weights are extracted through the legacy state_dict keys so this test
    works for both the old and the new implementation.
    """
    for num_experts, num_inputs in ((4, 10), (2, 3), (3, 3)):
        cfg, model = _make_tiny_model(
            num_experts=num_experts, seed=1000 + num_experts,
            moe_capacity_factor=1.25,
        )
        mlp = model.transformer.h[0].mlp
        gate_list, up_list, down_list, router_w = _extract_moe_weights(mlp)

        torch.manual_seed(2000 + num_experts)
        for i in range(num_inputs):
            x = torch.randn(2, 8, cfg.n_embd)
            out, aux = mlp(x)
            ref_out, ref_aux = _reference_moe_forward(
                x, gate_list, up_list, down_list, router_w,
                cfg.moe_capacity_factor, cfg.moe_aux_loss_factor,
            )
            assert torch.allclose(out, ref_out, atol=1e-5, rtol=0), (
                f"E={num_experts}, input {i}: output mismatch "
                f"(max abs diff {(out - ref_out).abs().max().item():.3e})"
            )
            assert torch.allclose(aux, ref_aux, atol=1e-5, rtol=0), (
                f"E={num_experts}, input {i}: aux_loss mismatch"
            )


# ---------------------------------------------------------------------------
# 4. Capacity int() truncation semantics (pin)
# ---------------------------------------------------------------------------

def test_capacity_int_truncation_semantics():
    """capacity must be int(cf * N / E) with truncation, NOT ceil.

    N=10 tokens, E=4, cf=1.0 -> int(2.5) = 2. All tokens are routed to
    expert 0 (router row 0 = ones, other rows = 0, strictly positive input),
    so exactly the first 2 tokens survive and tokens 2..9 are dropped. A ceil
    implementation (cap=3) would keep token 2 and fail this test.
    """
    cfg, model = _make_tiny_model(num_experts=4, seed=31337,
                                  moe_capacity_factor=1.0)
    mlp = model.transformer.h[0].mlp
    C = cfg.n_embd

    # force every token onto expert 0
    with torch.no_grad():
        mlp.router.weight.zero_()
        mlp.router.weight[0].add_(1.0)

    torch.manual_seed(31338)
    x = torch.rand(1, 10, C) * 2.0 + 0.5  # strictly positive -> expert 0 wins

    # sanity: the arithmetic under test really is 2.5 -> 2
    assert int(1.0 * 10 / 4) == 2

    out, aux = mlp(x)
    gate_list, up_list, down_list, router_w = _extract_moe_weights(mlp)
    ref_out, ref_aux = _reference_moe_forward(
        x, gate_list, up_list, down_list, router_w,
        cfg.moe_capacity_factor, cfg.moe_aux_loss_factor,
    )
    assert torch.allclose(out, ref_out, atol=1e-5, rtol=0), (
        "output must match the int()-truncation reference bit-for-bit"
    )
    assert torch.allclose(aux, ref_aux, atol=1e-6, rtol=0)

    row_norm = out[0].abs().sum(dim=-1)  # [T=10]
    assert row_norm[:2].min().item() > 0.0, (
        "first capacity=2 tokens must be processed (non-zero output)"
    )
    assert row_norm[2:].max().item() == 0.0, (
        "tokens beyond capacity=2 must be dropped exactly (int truncation, "
        "not ceil); a ceil implementation would keep 3 tokens"
    )


# ---------------------------------------------------------------------------
# 5. Dropped-token set matches reference (pin)
# ---------------------------------------------------------------------------

def test_capacity_dropped_tokens_match_reference():
    """cf=0.25 forces drops; the dropped set must match the reference exactly."""
    cfg, model = _make_tiny_model(num_experts=2, seed=4242,
                                  moe_capacity_factor=0.25)
    mlp = model.transformer.h[0].mlp
    # N=8, E=2, cf=0.25 -> capacity = int(0.25*8/2) = 1 per expert

    torch.manual_seed(4243)
    x = torch.randn(1, 8, cfg.n_embd)
    out, aux = mlp(x)
    gate_list, up_list, down_list, router_w = _extract_moe_weights(mlp)
    ref_out, ref_aux = _reference_moe_forward(
        x, gate_list, up_list, down_list, router_w,
        cfg.moe_capacity_factor, cfg.moe_aux_loss_factor,
    )

    # (a) full output equivalence
    assert torch.allclose(out, ref_out, atol=1e-5, rtol=0)
    assert torch.allclose(aux, ref_aux, atol=1e-6, rtol=0)

    # (b) zero-row (dropped) sets identical
    zero_model = out[0].abs().sum(dim=-1) == 0
    zero_ref = ref_out[0].abs().sum(dim=-1) == 0
    assert torch.equal(zero_model, zero_ref), (
        "dropped-token set must match the reference exactly"
    )

    # (c) at least one token actually dropped (test is not vacuous)
    assert zero_ref.sum().item() >= 1, (
        "cf=0.25 with N=8, E=2 must drop at least one token"
    )


# ---------------------------------------------------------------------------
# 6. Legacy checkpoint load equivalence (pin, B1)
# ---------------------------------------------------------------------------

def test_legacy_checkpoint_load_equivalence():
    """An old-format (ModuleList-key) checkpoint must load into a fresh model.

    The old-format dict is constructed manually with explicit
    gate_proj.{e}.weight / up_proj.{e}.weight / down_proj.{e}.weight /
    router.weight keys to simulate a checkpoint saved by the pre-T012
    implementation. strict=True must pass and the two models must produce
    bit-identical forwards.
    """
    cfg, model_a = _make_tiny_model(num_experts=4, seed=1234)
    _, model_b = _make_tiny_model(num_experts=4, seed=4321)

    sd_a = model_a.state_dict()
    E = cfg.num_experts
    prefix = "transformer.h.0.mlp."

    # manually build the old-format checkpoint: non-MoE entries copied,
    # MoE entries re-added with explicit ModuleList-style keys
    old_ckpt = {k: v.clone() for k, v in sd_a.items() if not k.startswith(prefix)}
    for e in range(E):
        old_ckpt[f"{prefix}gate_proj.{e}.weight"] = sd_a[f"{prefix}gate_proj.{e}.weight"].clone()
        old_ckpt[f"{prefix}up_proj.{e}.weight"] = sd_a[f"{prefix}up_proj.{e}.weight"].clone()
        old_ckpt[f"{prefix}down_proj.{e}.weight"] = sd_a[f"{prefix}down_proj.{e}.weight"].clone()
    old_ckpt[f"{prefix}router.weight"] = sd_a[f"{prefix}router.weight"].clone()

    model_b.load_state_dict(old_ckpt, strict=True)

    torch.manual_seed(777)
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    logits_a, _, _, aux_a = model_a(tokens, return_aux_loss=True)
    logits_b, _, _, aux_b = model_b(tokens, return_aux_loss=True)
    assert torch.equal(logits_a, logits_b), (
        "legacy checkpoint load must give bit-identical model forward"
    )
    assert torch.equal(aux_a, aux_b)


# ---------------------------------------------------------------------------
# 7. state_dict saves legacy keys + round-trip (pin, B1)
# ---------------------------------------------------------------------------

def test_state_dict_saves_legacy_keys():
    """state_dict() must keep the on-disk legacy key format exactly.

    Keys: router.weight + gate_proj.{e}.weight / up_proj.{e}.weight /
    down_proj.{e}.weight for e in 0..E-1, with per-expert [H,C]/[C,H] shapes.
    No gate_w / up_w / down_w keys may leak. Round-trip into a fresh model
    must preserve the forward bit-for-bit.
    """
    cfg, model = _make_tiny_model(num_experts=4, seed=2024)
    mlp = model.transformer.h[0].mlp
    E, H, C = cfg.num_experts, cfg.intermediate_size, cfg.n_embd

    sd = mlp.state_dict()
    expected = {"router.weight"}
    for e in range(E):
        expected.add(f"gate_proj.{e}.weight")
        expected.add(f"up_proj.{e}.weight")
        expected.add(f"down_proj.{e}.weight")

    assert set(sd.keys()) == expected, (
        f"state_dict keys must be exactly the legacy set; got {sorted(sd.keys())}"
    )
    for bad in ("gate_w", "up_w", "down_w"):
        assert all(bad not in k for k in sd.keys()), (
            f"stacked-parameter key '{bad}' must not appear in state_dict"
        )
    assert tuple(sd["gate_proj.0.weight"].shape) == (H, C)
    assert tuple(sd["up_proj.0.weight"].shape) == (H, C)
    assert tuple(sd["down_proj.0.weight"].shape) == (C, H)
    assert tuple(sd["router.weight"].shape) == (E, C)

    # round-trip through a fresh, differently-initialized model
    torch.manual_seed(999)
    model2 = ModernGPT(cfg)
    model2.eval()
    model2.load_state_dict(model.state_dict(), strict=True)

    torch.manual_seed(555)
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    logits_a, _, _, aux_a = model(tokens, return_aux_loss=True)
    logits_b, _, _, aux_b = model2(tokens, return_aux_loss=True)
    assert torch.equal(logits_a, logits_b)
    assert torch.equal(aux_a, aux_b)


# ---------------------------------------------------------------------------
# 8. num_experts <= 1 dense path unchanged (pin)
# ---------------------------------------------------------------------------

def test_num_experts_one_dense_path_unchanged():
    """num_experts=1 must keep the plain dense SwiGLU path and zero aux loss."""
    cfg, model = _make_tiny_model(num_experts=1, seed=88)
    mlp = model.transformer.h[0].mlp

    assert isinstance(mlp.gate_proj, nn.Linear)
    assert isinstance(mlp.up_proj, nn.Linear)
    assert isinstance(mlp.down_proj, nn.Linear)

    torch.manual_seed(89)
    x = torch.randn(2, 8, cfg.n_embd)
    out, aux = mlp(x)
    assert aux.item() == 0.0

    g = F.silu(F.linear(x, mlp.gate_proj.weight))
    u = F.linear(x, mlp.up_proj.weight)
    ref = F.linear(g * u, mlp.down_proj.weight)
    assert torch.allclose(out, ref, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# 9. aux loss identical (pin)
# ---------------------------------------------------------------------------

def test_aux_loss_identical():
    """aux_loss must stay on the exact same formula as the reference."""
    for cf in (1.25, 0.5):
        cfg, model = _make_tiny_model(num_experts=4, seed=606,
                                      moe_capacity_factor=cf)
        mlp = model.transformer.h[0].mlp
        gate_list, up_list, down_list, router_w = _extract_moe_weights(mlp)

        torch.manual_seed(607)
        for i in range(10):
            x = torch.randn(2, 8, cfg.n_embd)
            _, aux = mlp(x)
            _, ref_aux = _reference_moe_forward(
                x, gate_list, up_list, down_list, router_w,
                cfg.moe_capacity_factor, cfg.moe_aux_loss_factor,
            )
            assert torch.allclose(aux, ref_aux, atol=1e-6, rtol=0), (
                f"cf={cf}, input {i}: aux_loss mismatch "
                f"({aux.item():.8f} vs {ref_aux.item():.8f})"
            )


# ---------------------------------------------------------------------------
# 10. capacity == 0 boundary (pin)
# ---------------------------------------------------------------------------

def test_capacity_zero_all_dropped():
    """E=8, N=2, cf=1.25 -> capacity = int(0.3125) = 0: everything dropped.

    Forward must not crash, output must be all zeros, aux_loss must be finite.
    """
    cfg, model = _make_tiny_model(num_experts=8, seed=707,
                                  moe_capacity_factor=1.25)
    mlp = model.transformer.h[0].mlp
    assert int(1.25 * 2 / 8) == 0  # sanity of the arithmetic

    torch.manual_seed(708)
    x = torch.randn(1, 2, cfg.n_embd)
    out, aux = mlp(x)

    assert out.shape == (1, 2, cfg.n_embd)
    assert torch.equal(out, torch.zeros_like(out)), (
        "with capacity=0 every token is dropped; output must be exactly zero"
    )
    assert torch.isfinite(aux)


# ---------------------------------------------------------------------------
# 11. Speedup benchmark (Red: new implementation must be >= 2x faster)
# ---------------------------------------------------------------------------

def test_moe_speedup_at_least_2x():
    """MoE fwd+bwd must be at least 2x faster than the loop-based reference.

    Benchmark scale: E=4, H=688, C=256, N=512 (B=2, T=256), fwd+bwd
    (grads w.r.t. input and all weights), 3 warmup + 10 reps, median.
    """
    E, H, C, N = 4, 688, 256, 512
    gate_w = _uniform_linear_weight((E, H, C), C, seed=31415)
    up_w = _uniform_linear_weight((E, H, C), C, seed=31416)
    down_w = _uniform_linear_weight((E, C, H), H, seed=31417)
    router_w = _uniform_linear_weight((E, C), C, seed=31418)

    # new implementation: the model's MoE layer, weights loaded via the
    # legacy state_dict key format
    cfg = ModernGPTConfig(
        n_layer=1,
        n_head=4,
        n_embd=C,
        block_size=256,
        intermediate_size=H,
        num_experts=E,
        dropout=0.0,
        moe_aux_loss_factor=0.01,
        moe_capacity_factor=1.25,
    )
    mlp = SwiGLU(cfg)
    mlp.eval()
    sd = {}
    for e in range(E):
        sd[f"gate_proj.{e}.weight"] = gate_w[e].detach()
        sd[f"up_proj.{e}.weight"] = up_w[e].detach()
        sd[f"down_proj.{e}.weight"] = down_w[e].detach()
    sd["router.weight"] = router_w.detach()
    mlp.load_state_dict(sd, strict=True)

    torch.manual_seed(31419)
    x = torch.randn(2, N // 2, C).requires_grad_(True)  # [B=2, T=256, C]

    gate_list = [gate_w[e] for e in range(E)]
    up_list = [up_w[e] for e in range(E)]
    down_list = [down_w[e] for e in range(E)]

    def run_old():
        out, _ = _reference_moe_forward(
            x, gate_list, up_list, down_list, router_w, 1.25, 0.01
        )
        loss = out.square().mean()
        torch.autograd.grad(
            loss, [x, gate_w, up_w, down_w, router_w], allow_unused=True
        )

    new_params = [x] + list(mlp.parameters())

    def run_new():
        out, _ = mlp(x)
        loss = out.square().mean()
        torch.autograd.grad(loss, new_params, allow_unused=True)

    old_ms = _median_ms(run_old, warmup=3, reps=10)
    new_ms = _median_ms(run_new, warmup=3, reps=10)
    ratio = old_ms / new_ms

    assert ratio >= 2.0, (
        f"MoE fwd+bwd speedup {ratio:.2f}x < 2.0x "
        f"(old reference {old_ms:.1f} ms/step, new {new_ms:.1f} ms/step)"
    )


# ---------------------------------------------------------------------------
# 12. torch.compile recompile stability (skip on Windows inductor failure)
# ---------------------------------------------------------------------------

def test_compile_no_recompile_growth():
    """100 identical steps through a compiled MoE layer must not recompile."""
    try:
        probe = torch.compile(lambda t: t * 2)
        result = probe(torch.ones(4))
        assert torch.allclose(result, torch.full((4,), 2.0))
    except Exception:
        pytest.skip("torch.compile unavailable on this platform")

    torch._dynamo.reset()
    cfg, model = _make_tiny_model(num_experts=4, seed=1212)
    mlp = model.transformer.h[0].mlp
    torch.manual_seed(1213)
    x = torch.randn(2, 8, cfg.n_embd)

    compiled = torch.compile(mlp)
    for _ in range(3):  # warmup: pay all compilation costs up front
        compiled(x)

    counters = torch._dynamo.utils.counters
    before = counters["stats"]["unique_graphs"]
    for _ in range(100):
        compiled(x)
    after = counters["stats"]["unique_graphs"]

    assert after == before, (
        f"recompile count grew over 100 identical steps: {before} -> {after}"
    )
