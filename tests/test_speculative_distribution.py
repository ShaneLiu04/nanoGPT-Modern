"""Red-phase tests: sampling / speculative-decoding distribution defects
(AR-v0.5.0-hpc-readiness T005 / FR4).

Target defects these tests must expose:

1. ``temperature=0`` must be greedy (argmax) in all three sampling entries:
   - model/modern_gpt.py ``generate`` ``_sample`` closure (~L1644)
   - inference/generate_utils.py ``_sample_logits`` (L20-43)
   - inference/continuous_batching.py ``BatchGenerator._sample`` (L210-234)
   Currently all three fall through to ``torch.multinomial`` at
   temperature=0, i.e. sampling is random.
2. Speculative decoding with draft==target at temperature=0 must match direct
   greedy generation.
3. design.md §4.3.5 (L344-348) requires extracted module-level interfaces
   ``_draft_sample`` and ``_verify`` in model/modern_gpt.py with corrected
   distributions:
   - greedy draft p_d is a point mass (selected-token prob == 1.0), not the
     argmax token's softmax probability;
   - rejection residual q = norm(relu(p_t_tempered - q_draft_tempered)) using
     the *complete* tempered draft distribution;
   - verification softmax is tempered with the same temperature as the final
     sampling;
   - draft/target vocab mismatch raises ValueError.
4. Statistical equivalence: with draft != target at temperature=1.0, the
   speculative output token distribution must match direct sampling (the
   current p_d bug inflates the acceptance ratio and skews the distribution
   toward the draft model's argmax).
"""

import collections

import pytest
import torch

from model.modern_gpt import ModernGPT, ModernGPTConfig


def _tiny_model(seed=0, vocab_size=100, n_layer=2, n_embd=32, block_size=64):
    torch.manual_seed(seed)
    cfg = ModernGPTConfig(
        n_layer=n_layer,
        n_head=2,
        n_embd=n_embd,
        block_size=block_size,
        vocab_size=vocab_size,
        dropout=0.0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ModernGPT(cfg).to(device).eval()


# ---------------------------------------------------------------------------
# 1. temperature=0 determinism (three sampling entries)
# ---------------------------------------------------------------------------


def test_generate_temperature_zero_is_deterministic():
    """modern_gpt.generate(temperature=0): two runs must be bit-identical and
    equal to greedy (top_k=1) decoding.  Currently temperature=0 skips the
    temperature branch and samples via multinomial → random output."""
    model = _tiny_model(seed=11)
    device = next(model.parameters()).device
    idx = torch.tensor([[3, 17, 5, 42]], device=device)

    with torch.no_grad():
        out1 = model.generate(
            idx.clone(), max_new_tokens=32, use_cache=True,
            temperature=0.0, top_k=None,
        )
        out2 = model.generate(
            idx.clone(), max_new_tokens=32, use_cache=True,
            temperature=0.0, top_k=None,
        )
        greedy = model.generate(
            idx.clone(), max_new_tokens=32, use_cache=True, top_k=1
        )

    assert torch.equal(out1, out2), (
        "generate(temperature=0) must be deterministic (argmax); "
        "two runs produced different outputs"
    )
    assert torch.equal(out1, greedy), (
        "generate(temperature=0) must equal greedy (top_k=1) decoding"
    )


class _FakeTokenizer:
    """Minimal tokenizer for batched_generate (ids are whitespace-separated)."""

    eot_token = 999999

    def encode(self, s):
        return [int(t) for t in s.split()]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def test_generate_utils_temperature_zero_is_deterministic():
    """generate_utils.batched_generate(temperature=0): two runs must be
    bit-identical and equal to greedy (top_k=1).  _sample_logits currently
    falls through to multinomial at temperature=0 → random output."""
    from inference.generate_utils import batched_generate

    model = _tiny_model(seed=23)
    device = next(model.parameters()).device
    tok = _FakeTokenizer()
    prompts = ["3 17", "5 42 7"]

    with torch.no_grad():
        _, ids1 = batched_generate(
            model, prompts, tok, max_new_tokens=12, device=device,
            temperature=0.0, top_k=None, eos_token_id=99, return_ids=True,
        )
        _, ids2 = batched_generate(
            model, prompts, tok, max_new_tokens=12, device=device,
            temperature=0.0, top_k=None, eos_token_id=99, return_ids=True,
        )
        _, ids_g = batched_generate(
            model, prompts, tok, max_new_tokens=12, device=device,
            top_k=1, eos_token_id=99, return_ids=True,
        )

    assert ids1 == ids2, (
        "batched_generate(temperature=0) must be deterministic (argmax); "
        "two runs produced different outputs"
    )
    assert ids1 == ids_g, (
        "batched_generate(temperature=0) must equal greedy (top_k=1) decoding"
    )


def test_continuous_batching_sample_temperature_zero_is_greedy():
    """continuous_batching.BatchGenerator._sample(temperature=0): repeated
    calls on fixed logits must all return the argmax token.  Currently
    temperature=0 skips the temperature branch and samples via multinomial →
    random output."""
    from inference.continuous_batching import BatchGenerator

    model = _tiny_model(seed=31)
    gen = BatchGenerator(model, device="cpu")
    torch.manual_seed(3)
    logits = torch.randn(1, 100)
    expected = logits.argmax(dim=-1).item()

    outs = [
        gen._sample(logits, temperature=0.0, top_k=None, top_p=None).item()
        for _ in range(16)
    ]

    assert all(o == outs[0] for o in outs), (
        "BatchGenerator._sample(temperature=0) must be deterministic "
        "(argmax); repeated calls returned different tokens"
    )
    assert outs[0] == expected, (
        "BatchGenerator._sample(temperature=0) must return the argmax token"
    )


# ---------------------------------------------------------------------------
# 2. Speculative decoding, draft == target, temperature=0
# ---------------------------------------------------------------------------


def test_speculative_self_draft_temperature_zero_matches_direct():
    """With the target model as its own draft, speculative decoding at
    temperature=0 must reproduce direct generate(temperature=0) output
    bit-for-bit (greedy draft + point-mass verification accepts exactly the
    greedy tokens)."""
    model = _tiny_model(seed=11)
    device = next(model.parameters()).device
    idx = torch.tensor([[3, 17, 5, 42]], device=device)

    with torch.no_grad():
        direct = model.generate(
            idx.clone(), max_new_tokens=12, use_cache=True,
            temperature=0.0, top_k=None,
        )
        spec = model.generate(
            idx.clone(), max_new_tokens=12, use_cache=True,
            temperature=0.0, top_k=None,
            draft_model=model, draft_tokens=4,
        )

    assert torch.equal(direct, spec), (
        "speculative decoding with draft==target at temperature=0 must match "
        "direct greedy generation"
    )


# ---------------------------------------------------------------------------
# 3. Extracted interfaces: _draft_sample / _verify (design.md §4.3.5)
# ---------------------------------------------------------------------------


def _selected_prob(p_d, toks, V, B):
    """Extract the selected-token probability from _draft_sample's p_d return.

    Accepts either the per-selected-token probability vector [B] or the full
    tempered draft distribution [B, V].
    """
    assert torch.is_tensor(p_d), "p_d returned by _draft_sample must be a Tensor"
    if p_d.dim() == 2 and p_d.shape[-1] == V and p_d.shape[0] == B:
        idx = torch.tensor(toks, dtype=torch.long).view(B, 1)
        return p_d.gather(1, idx).reshape(-1)
    flat = p_d.reshape(-1).float()
    assert flat.numel() == B, (
        f"cannot interpret p_d shape {tuple(p_d.shape)} for batch {B}, vocab {V}"
    )
    return flat


def _flatten_tokens(accepted):
    """Flatten _verify's returned token sequence to a plain int list."""
    if torch.is_tensor(accepted):
        return [int(x) for x in accepted.reshape(-1).tolist()]
    if isinstance(accepted, (list, tuple)):
        out = []
        for t in accepted:
            if torch.is_tensor(t):
                out.extend(int(x) for x in t.reshape(-1).tolist())
            else:
                out.append(int(t))
        return out
    raise TypeError(f"unsupported accepted-tokens type: {type(accepted)}")


def test_draft_sample_greedy_point_mass():
    """_draft_sample(logits, temperature=0) must return (token, p_d) with
    token == argmax and p_d == point mass (selected-token probability 1.0).

    The current inline draft logic returns the argmax token's *softmax*
    probability (≈0.02 for a flat distribution), which inflates the
    speculative acceptance ratio."""
    try:
        from model.modern_gpt import _draft_sample
    except ImportError as e:
        pytest.fail(f"design §4.3.5 interface _draft_sample not extracted: {e}")

    torch.manual_seed(5)
    B, V = 2, 6  # B != V so return-shape dispatch is unambiguous
    logits = torch.randn(B, V)

    ret = _draft_sample(logits, temperature=0.0)
    assert isinstance(ret, (tuple, list)) and len(ret) == 2, (
        "_draft_sample must return (token, p_d)"
    )
    tok, p_d = ret
    toks = torch.as_tensor(tok).reshape(-1).tolist()
    assert len(toks) == B
    am = logits.argmax(dim=-1).tolist()
    assert toks == am, "greedy draft (temperature=0) must select the argmax token"

    sel = _selected_prob(p_d, toks, V, B)
    assert torch.allclose(sel, torch.ones(B), atol=1e-6), (
        "greedy draft p_d must be a point mass: the selected token's "
        "probability must be 1.0, not its softmax probability"
    )


def test_draft_sample_tempered_probabilities():
    """_draft_sample at temperature>0 must use the tempered softmax:
    p_d(selected) == softmax(logits / temperature)[selected]."""
    try:
        from model.modern_gpt import _draft_sample
    except ImportError as e:
        pytest.fail(f"design §4.3.5 interface _draft_sample not extracted: {e}")

    torch.manual_seed(6)
    B, V = 2, 6
    logits = torch.randn(B, V)

    for T in (1.0, 0.5, 2.0):
        ret = _draft_sample(logits, temperature=T)
        assert isinstance(ret, (tuple, list)) and len(ret) == 2
        tok, p_d = ret
        toks = torch.as_tensor(tok).reshape(-1).tolist()
        assert len(toks) == B
        assert all(0 <= t < V for t in toks), "draft token out of vocab range"

        sel = _selected_prob(p_d, toks, V, B)
        tempered = torch.softmax(logits / T, dim=-1)
        exp_sel = tempered.gather(
            1, torch.tensor(toks, dtype=torch.long).view(B, 1)
        ).reshape(-1)
        assert torch.allclose(sel, exp_sel.float(), atol=1e-5), (
            f"_draft_sample p_d at temperature={T} must be the tempered "
            "softmax probability softmax(logits/T) of the selected token"
        )


def test_verify_forced_rejection_residual_is_target_minus_draft():
    """_verify rejection path: residual q = norm(relu(p_t - q_draft)) over the
    FULL draft distribution, then sample the replacement from q.

    Construction (temperature=1.0, V=4): greedy draft proposes token 1 with
    point-mass p_d; the target distribution is a point mass at token 2, so
    the acceptance ratio is 0 and rejection is forced.  The residual
    relu(point_mass_2 - point_mass_1) is a point mass at token 2, so the
    replacement token must be deterministically 2 and n_accepted == 0."""
    try:
        from model.modern_gpt import _verify
    except ImportError as e:
        pytest.fail(f"design §4.3.5 interface _verify not extracted: {e}")

    V = 4
    draft_tokens = torch.tensor([[1]])
    draft_probs = torch.zeros(1, 1, V)
    draft_probs[0, 0, 1] = 1.0  # greedy draft: point mass at token 1
    target_logits = torch.full((1, 1, V), -1e9)
    target_logits[0, 0, 2] = 0.0  # target: point mass at token 2

    result = _verify(draft_tokens, target_logits, draft_probs, temperature=1.0)
    assert isinstance(result, (tuple, list)) and len(result) == 2, (
        "_verify must return (accepted_tokens, n_accepted)"
    )
    accepted, n_accepted = result

    assert int(n_accepted) == 0, "acceptance ratio is 0 here; token 1 must be rejected"
    toks = _flatten_tokens(accepted)
    assert toks == [2], (
        "rejection replacement must be sampled from "
        "norm(relu(p_t - q_draft)); with these inputs that residual is a "
        f"point mass at token 2, got {toks}"
    )


def test_verify_accept_then_reject_multi_position():
    """_verify over gamma=2: position 0 accepted (ratio 1), position 1
    rejected (ratio 0) with a deterministic residual replacement.

    Contract pinned by this test: the returned token sequence contains the
    accepted draft prefix followed by the resampled replacement token, while
    n_accepted counts only the accepted draft tokens."""
    try:
        from model.modern_gpt import _verify
    except ImportError as e:
        pytest.fail(f"design §4.3.5 interface _verify not extracted: {e}")

    V = 4
    draft_tokens = torch.tensor([[2, 1]])
    draft_probs = torch.zeros(1, 2, V)
    draft_probs[0, 0, 2] = 1.0  # pos 0: draft point mass at token 2
    draft_probs[0, 1, 1] = 1.0  # pos 1: draft point mass at token 1
    target_logits = torch.full((1, 2, V), -1e9)
    target_logits[0, 0, 2] = 0.0  # pos 0: target point mass at 2 → ratio 1 → accept
    target_logits[0, 1, 3] = 0.0  # pos 1: target point mass at 3 → ratio 0 → reject

    accepted, n_accepted = _verify(draft_tokens, target_logits, draft_probs, 1.0)

    assert int(n_accepted) == 1, "exactly the first draft token must be accepted"
    toks = _flatten_tokens(accepted)
    assert toks[0] == 2, "the accepted draft token (2) must come first"
    assert len(toks) == 2 and toks[1] == 3, (
        "after a rejection the sequence must include the replacement token "
        "sampled from the residual (a point mass at token 3 here), "
        f"got {toks}"
    )


def test_verify_residual_and_acceptance_use_temperature():
    """_verify must temper the target softmax with ``temperature``.

    Construction (temperature=0.5, V=4): target_logits = 0.5*log(u) so that
    softmax(target_logits / 0.5) == u = [0.3, 1e-9, 0.6, 0.1].  The greedy
    draft proposes token 1 with p_d=0.5 → acceptance ratio ≈ 2e-9 → rejection
    is forced.  The tempered residual relu(u - d) with d = [0.3, 0.5, 0.1,
    0.1] is a point mass at token 2, so the replacement is deterministically
    2.  If _verify skips the tempering (bug: untempered softmax), the residual
    also places mass on tokens 0 and 3 and the replacement varies."""
    try:
        from model.modern_gpt import _verify
    except ImportError as e:
        pytest.fail(f"design §4.3.5 interface _verify not extracted: {e}")

    V = 4
    T = 0.5
    u = torch.tensor([0.3, 1e-9, 0.6, 0.1])
    target_logits = (T * torch.log(u)).view(1, 1, V)
    draft_tokens = torch.tensor([[1]])
    draft_probs = torch.tensor([0.3, 0.5, 0.1, 0.1]).view(1, 1, V)

    torch.manual_seed(0)
    for trial in range(40):
        accepted, n_accepted = _verify(
            draft_tokens, target_logits, draft_probs, temperature=T
        )
        assert int(n_accepted) == 0, (
            f"trial {trial}: acceptance ratio must be p_t(1)/p_d(1) ≈ 2e-9 "
            "(tempered), so rejection is forced"
        )
        toks = _flatten_tokens(accepted)
        assert toks == [2], (
            f"trial {trial}: the tempered residual "
            "norm(relu(softmax(target_logits/T) - draft_probs)) is a point "
            "mass at token 2; an untempered verification softmax would also "
            f"place mass on tokens 0 and 3 (got {toks})"
        )


def test_verify_vocab_mismatch_raises_valueerror():
    """_verify must raise ValueError when draft and target vocab sizes differ
    (design §4.3.5, B6)."""
    try:
        from model.modern_gpt import _verify
    except ImportError as e:
        pytest.fail(f"design §4.3.5 interface _verify not extracted: {e}")

    # draft_probs vocab (3) != target_logits vocab (4)
    with pytest.raises(ValueError):
        _verify(
            torch.tensor([[1]]),
            torch.zeros(1, 1, 4),
            torch.zeros(1, 1, 3),
            temperature=1.0,
        )
    # draft token id outside the target vocab
    with pytest.raises(ValueError):
        _verify(
            torch.tensor([[9]]),
            torch.zeros(1, 1, 4),
            torch.zeros(1, 1, 4),
            temperature=1.0,
        )


# ---------------------------------------------------------------------------
# 4. Statistical equivalence: draft != target, temperature=1.0
# ---------------------------------------------------------------------------


def test_speculative_matches_direct_sampling_distribution():
    """Speculative decoding must be distribution-preserving.

    With a draft model different from the target (same architecture,
    different init) at temperature=1.0, the first generated token's
    empirical distribution under speculative generate() must match direct
    sampling (total variation distance < 0.25 over n=300 samples; a correct
    implementation measures ≈0.07 on this setup, while the current p_d bug —
    argmax softmax probability instead of a greedy point mass — clamps the
    acceptance ratio to 1 and emits the draft model's argmax nearly always,
    giving TVD ≈ 0.8)."""
    V = 8
    target = _tiny_model(seed=1234, vocab_size=V, n_embd=16)
    draft = _tiny_model(seed=5678, vocab_size=V, n_embd=16)
    device = next(target.parameters()).device
    prompt = torch.tensor([[3, 1, 5, 2]], device=device)

    n = 300
    direct_counts = collections.Counter()
    spec_counts = collections.Counter()
    with torch.no_grad():
        for _ in range(n):
            t_direct = target.generate(
                prompt.clone(), max_new_tokens=1, use_cache=True,
                temperature=1.0, top_k=None,
            )[0, -1].item()
            t_spec = target.generate(
                prompt.clone(), max_new_tokens=1, use_cache=True,
                temperature=1.0, top_k=None,
                draft_model=draft, draft_tokens=4,
            )[0, -1].item()
            direct_counts[t_direct] += 1
            spec_counts[t_spec] += 1

    tvd = 0.5 * sum(
        abs(spec_counts.get(t, 0) - direct_counts.get(t, 0)) / n
        for t in range(V)
    )
    assert tvd < 0.25, (
        "speculative decoding (temperature=1.0) is not distribution-preserving: "
        f"TVD vs direct sampling = {tvd:.3f} "
        f"(spec counts={dict(spec_counts)}, direct counts={dict(direct_counts)}); "
        "the acceptance ratio must use the greedy point-mass p_d, not the "
        "argmax token's softmax probability"
    )
