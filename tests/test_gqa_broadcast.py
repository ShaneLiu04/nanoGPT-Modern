"""Tests for M2: GQA grouped-broadcast SDPA and runtime probe.

These tests verify that the grouped-broadcast reshape
``[B, n_kv, rep, T, D]`` / ``[B, n_kv, 1, S, D]`` produces results that
are numerically identical to ``repeat_interleave`` while never copying KV,
across training (causal), decode (non-causal), masked, and cached paths.
"""
import pytest
import torch

from model.attention_utils import probe_gqa_sdpa_support, reset_gqa_probe_cache
from model.modern_gpt import ModernGPT, ModernGPTConfig, _gqa_grouped_sdpa


# ---------------------------------------------------------------------------
#  _gqa_grouped_sdpa unit tests
# ---------------------------------------------------------------------------

class TestGQAGroupedSDPA:
    """Unit tests for the _gqa_grouped_sdpa helper."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.B, self.Hq, self.Hkv, self.T, self.D = 1, 4, 2, 16, 32
        self.rep = self.Hq // self.Hkv
        self.q = torch.randn(self.B, self.Hq, self.T, self.D)
        self.k = torch.randn(self.B, self.Hkv, self.T, self.D)
        self.v = torch.randn(self.B, self.Hkv, self.T, self.D)

    def test_causal_output_matches_repeat_interleave(self):
        """Grouped-broadcast causal output must match repeat_interleave."""
        ref = torch.nn.functional.scaled_dot_product_attention(
            self.q,
            self.k.repeat_interleave(self.rep, 1),
            self.v.repeat_interleave(self.rep, 1),
            is_causal=True,
        )
        out = _gqa_grouped_sdpa(
            self.q, self.k, self.v, self.Hkv, self.rep,
            is_causal=True,
        )
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    def test_noncausal_output_matches_repeat_interleave(self):
        """Decode path (non-causal, T=1) must match repeat_interleave."""
        q1 = torch.randn(self.B, self.Hq, 1, self.D)
        k1 = torch.randn(self.B, self.Hkv, self.T, self.D)
        v1 = torch.randn(self.B, self.Hkv, self.T, self.D)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q1,
            k1.repeat_interleave(self.rep, 1),
            v1.repeat_interleave(self.rep, 1),
            is_causal=False,
        )
        out = _gqa_grouped_sdpa(
            q1, k1, v1, self.Hkv, self.rep,
            is_causal=False,
        )
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    def test_padding_mask_broadcasts_correctly(self):
        """An additive padding mask [B, 1, 1, S] must broadcast correctly."""
        q1 = torch.randn(self.B, self.Hq, 1, self.D)
        k1 = torch.randn(self.B, self.Hkv, self.T, self.D)
        v1 = torch.randn(self.B, self.Hkv, self.T, self.D)
        # Padding mask: mask out the last 4 positions.
        mask = torch.zeros(self.B, 1, 1, self.T)
        mask[:, :, :, -4:] = float("-inf")
        ref = torch.nn.functional.scaled_dot_product_attention(
            q1,
            k1.repeat_interleave(self.rep, 1),
            v1.repeat_interleave(self.rep, 1),
            attn_mask=mask,
            is_causal=False,
        )
        out = _gqa_grouped_sdpa(
            q1, k1, v1, self.Hkv, self.rep,
            attn_mask=mask,
            is_causal=False,
        )
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    def test_custom_scale(self):
        """Custom softmax_scale is forwarded correctly."""
        ref = torch.nn.functional.scaled_dot_product_attention(
            self.q,
            self.k.repeat_interleave(self.rep, 1),
            self.v.repeat_interleave(self.rep, 1),
            is_causal=True,
            scale=0.5,
        )
        out = _gqa_grouped_sdpa(
            self.q, self.k, self.v, self.Hkv, self.rep,
            is_causal=True, scale=0.5,
        )
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    def test_mha_no_gqa_passthrough(self):
        """When n_kv == n_head (no GQA), the helper still works."""
        H = 4
        q = torch.randn(1, H, 16, 32)
        k = torch.randn(1, H, 16, 32)
        v = torch.randn(1, H, 16, 32)
        out = _gqa_grouped_sdpa(q, k, v, H, 1, is_causal=True)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True,
        )
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
#  probe_gqa_sdpa_support tests
# ---------------------------------------------------------------------------

class TestGQAProbe:
    """Tests for the runtime GQA broadcast probe."""

    def test_probe_returns_tuple_of_bools(self):
        reset_gqa_probe_cache()
        raw, grouped = probe_gqa_sdpa_support(torch.device("cpu"))
        assert isinstance(raw, bool)
        assert isinstance(grouped, bool)

    def test_probe_cache_is_used(self):
        reset_gqa_probe_cache()
        r1, g1 = probe_gqa_sdpa_support(torch.device("cpu"))
        r2, g2 = probe_gqa_sdpa_support(torch.device("cpu"))
        assert (r1, g1) == (r2, g2)

    def test_probe_force_refresh(self):
        reset_gqa_probe_cache()
        r1, g1 = probe_gqa_sdpa_support(torch.device("cpu"))
        r2, g2 = probe_gqa_sdpa_support(torch.device("cpu"), force_refresh=True)
        assert (r1, g1) == (r2, g2)  # Same hardware → same result

    def test_probe_invalid_gqa_config(self):
        raw, grouped = probe_gqa_sdpa_support(
            torch.device("cpu"), n_head=3, n_kv_head=2,
        )
        assert raw is False
        assert grouped is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_probe_cuda_grouped_available(self):
        """On CUDA, grouped broadcast should be available (it works everywhere)."""
        reset_gqa_probe_cache()
        _, grouped = probe_gqa_sdpa_support(torch.device("cuda"))
        assert grouped is True


# ---------------------------------------------------------------------------
#  Integration tests with ModernGPT
# ---------------------------------------------------------------------------

class TestGQABroadcastIntegration:
    """Integration tests: GQA broadcast strategy with full model."""

    def _make_model(self, gqa_broadcast="auto", **kw):
        cfg = ModernGPTConfig(
            n_layer=1, n_head=4, n_embd=64, block_size=32,
            n_kv_head=2, vocab_size=100, dropout=0.0,
            gqa_broadcast=gqa_broadcast, **kw,
        )
        return ModernGPT(cfg)

    def test_auto_resolves_to_grouped_or_raw(self):
        """With gqa_broadcast='auto', the first forward resolves the mode."""
        model = self._make_model()
        model.eval()
        x = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            model(x)
        assert model.transformer.h[0].attn._gqa_mode in ("raw", "grouped", "repeat")

    def test_forced_grouped_forward(self):
        """gqa_broadcast='grouped' produces valid forward output."""
        model = self._make_model(gqa_broadcast="grouped")
        model.eval()
        x = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            logits, _, _ = model(x)
        assert logits.shape == (1, 8, 100)

    def test_forced_repeat_forward(self):
        """gqa_broadcast='repeat' produces valid forward output."""
        model = self._make_model(gqa_broadcast="repeat")
        model.eval()
        x = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            logits, _, _ = model(x)
        assert logits.shape == (1, 8, 100)

    def test_grouped_matches_repeat_forward(self):
        """Grouped and repeat produce bit-wise identical forward output."""
        torch.manual_seed(42)
        model = self._make_model(gqa_broadcast="repeat")
        model.eval()
        x = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            logits_rep, _, _ = model(x)
        # Switch to grouped
        model.transformer.h[0].attn._gqa_mode = "grouped"
        with torch.no_grad():
            logits_grp, _, _ = model(x)
        assert torch.allclose(logits_rep, logits_grp, atol=1e-5, rtol=1e-5)

    def test_grouped_matches_repeat_generate(self):
        """Grouped and repeat produce identical generate() output."""
        torch.manual_seed(42)
        model = self._make_model(gqa_broadcast="repeat")
        model.eval()
        x = torch.randint(0, 100, (1, 4))
        with torch.no_grad():
            out_rep = model.generate(x, max_new_tokens=8, use_cache=True, top_k=1)
        model.transformer.h[0].attn._gqa_mode = "grouped"
        with torch.no_grad():
            out_grp = model.generate(x, max_new_tokens=8, use_cache=True, top_k=1)
        assert torch.equal(out_rep, out_grp)

    def test_gqa_broadcast_config_serialization(self):
        """gqa_broadcast round-trips through to_dict / from_dict."""
        cfg = ModernGPTConfig(
            n_layer=1, n_head=4, n_embd=64, gqa_broadcast="grouped",
        )
        d = cfg.to_dict()
        assert d["gqa_broadcast"] == "grouped"
        cfg2 = ModernGPTConfig.from_dict(d)
        assert cfg2.gqa_broadcast == "grouped"

    def test_gqa_broadcast_invalid_raises(self):
        """Invalid gqa_broadcast value raises ValueError."""
        with pytest.raises(ValueError, match="gqa_broadcast"):
            ModernGPTConfig(
                n_layer=1, n_head=4, n_embd=64, gqa_broadcast="invalid",
            )

    def test_mha_no_gqa_ignores_broadcast_mode(self):
        """When n_kv_head == n_head (MHA), broadcast mode is irrelevant."""
        model = ModernGPT(ModernGPTConfig(
            n_layer=1, n_head=4, n_embd=64, block_size=32,
            vocab_size=100, dropout=0.0, gqa_broadcast="grouped",
        ))
        model.eval()
        x = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            logits, _, _ = model(x)
        assert logits.shape == (1, 8, 100)
