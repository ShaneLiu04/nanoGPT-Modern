"""Regression lock for the attention-mask + causal fold.

``CausalSelfAttention.forward`` can end up with an additive padding mask
while ``is_causal`` is still True (T > 1, no cache). PyTorch rejects that
combination -- 2.5.1 raises "Explicit attn_mask should not be set when
is_causal=True", newer versions changed the semantics -- so causality is
folded into the additive mask and ``is_causal`` is cleared.

These tests pin the fold's numerics: an all-ones mask must reproduce the
pure-causal fast path exactly, and right padding must not leak into the
outputs of fully-valid rows.
"""

import torch
from helpers import tiny_model


def test_all_ones_mask_matches_causal_fast_path():
    """Folding causality into an all-ones padding mask is a no-op."""
    model = tiny_model()
    torch.manual_seed(3)
    x = torch.randint(1, 100, (2, 10))

    logits_plain, _, _ = model(x)
    logits_masked, _, _ = model(x, attention_mask=torch.ones(2, 10, dtype=torch.bool))

    assert torch.allclose(
        logits_plain, logits_masked, atol=1e-5
    ), "all-ones attention_mask must not change causal forward outputs"
    print("[OK] all-ones attention_mask matches the causal fast path")


def test_right_padding_does_not_leak_into_valid_rows():
    """A fully-valid row keeps its solo-run outputs beside a padded row."""
    model = tiny_model()
    torch.manual_seed(4)
    seq = torch.randint(1, 100, (1, 8))

    logits_solo, _, _ = model(seq)

    x = torch.cat([seq, seq], dim=0)
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[1, 5:] = False  # row 1 right-padded from position 5

    logits_batch, _, _ = model(x, attention_mask=mask)

    assert torch.allclose(
        logits_batch[0], logits_solo[0], atol=1e-5
    ), "valid row outputs changed when batched with a padded row"
    assert torch.isfinite(logits_batch[1]).all()
    print("[OK] right padding does not leak into valid rows")
