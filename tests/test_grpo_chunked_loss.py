"""Regression locks for the large-batch GRPO paths (G*B > 256).

The cloud RTX-6000D runs exercise chunked code paths that the regular
suite never triggers with its tiny configs:

* ``_batch_logprobs_impl`` splits no-grad batches over 256 sequences and
  pads the per-chunk outputs back together.
* ``compute_grpo_loss`` splits the gradient graph into prompt chunks when
  ``group_size * batch_size`` exceeds ~256 sequences, backwards each chunk
  immediately and returns a detached logging scalar (the train loop then
  skips its own backward via ``loss.requires_grad``).

These tests pin both paths to the single-forward reference so future
refactors cannot silently change the numerical semantics.
"""

import numpy as np
import torch
from test_grpo import _make_tiny_checkpoint, _tiny_grpo_args

from training.train_grpo import GRPOTrainer

G, B = 4, 68  # 4 * 68 = 272 > 256 -> chunked paths trigger


def _make_trainer(tmpdir):
    ckpt = _make_tiny_checkpoint(tmpdir)
    args = _tiny_grpo_args(tmpdir, ckpt, group_size=G, batch_size=2, num_train=2, num_val=2)
    return GRPOTrainer(args)


def _build_rollout(trainer, seed=7):
    """Synthetic rollout with variable prompt/response lengths.

    Sequence totals range 4..7 tokens so different prompt chunks have
    different max lengths (exercising the ``old_full[:, :T]`` slicing in
    ``_grpo_surrogate``).
    """
    rng = np.random.default_rng(seed)
    prompt_tokens = []
    for b in range(B):
        p_len = 2 + (b % 2)
        prompt_tokens.append([11 + (b % 80) + i for i in range(p_len)])
    response_ids = [[None] * B for _ in range(G)]
    prompt_lens = [[0] * B for _ in range(G)]
    response_lens = [[0] * B for _ in range(G)]
    for g in range(G):
        for b in range(B):
            r_len = 2 + ((b + g) % 3)
            response_ids[g][b] = [31 + ((b * 7 + g * 13) % 60) + i for i in range(r_len)]
            prompt_lens[g][b] = len(prompt_tokens[b])
            response_lens[g][b] = r_len
    flat_seqs, flat_plens, flat_rlens = [], [], []
    for g in range(G):
        for b in range(B):
            flat_seqs.append(prompt_tokens[b] + response_ids[g][b])
            flat_plens.append(prompt_lens[g][b])
            flat_rlens.append(response_lens[g][b])

    model = trainer.model
    model.eval()
    # 272 > 256 under no_grad -> the old-logprob call itself takes the
    # chunked path; ref keeps the single-forward path (grad-guarded).
    with torch.no_grad():
        old_logprobs, masks = trainer._batch_logprobs(model, flat_seqs, flat_plens, flat_rlens)
    ref_logprobs, _ = trainer._batch_logprobs_ref(trainer.ref, flat_seqs, flat_plens, flat_rlens)

    rollout = {
        "responses": [["r"] * B for _ in range(G)],
        "response_ids": response_ids,
        "rewards": rng.normal(size=(G, B)).astype(np.float32),
        "old_logprobs": old_logprobs,
        "ref_logprobs": ref_logprobs,
        "masks": masks,
        "prompt_lens": prompt_lens,
        "response_lens": response_lens,
        "prompt_tokens": prompt_tokens,
    }
    return rollout, flat_seqs, flat_plens, flat_rlens


def test_batch_logprobs_splits_no_grad_batches_over_256(tmp_path):
    """B=300 under no_grad must equal the manually concatenated chunks."""
    trainer = _make_trainer(str(tmp_path))
    try:
        model = trainer.model
        model.eval()
        seqs, plens, rlens = [], [], []
        for i in range(300):
            # chunk 1 (i < 256) has totals 4..7; chunk 2 uniformly 4 so the
            # per-chunk max lengths differ and the padding path runs.
            r_len = 2 if i >= 256 else 2 + (i % 3)
            seqs.append([5 + (i % 90), 6 + (i % 90)] + [40 + ((i * 3) % 50) for _ in range(r_len)])
            plens.append(2)
            rlens.append(r_len)

        with torch.no_grad():
            lp_split, mask_split = trainer._batch_logprobs(model, seqs, plens, rlens)
            lp_a, mask_a = trainer._batch_logprobs(model, seqs[:256], plens[:256], rlens[:256])
            lp_b, mask_b = trainer._batch_logprobs(model, seqs[256:], plens[256:], rlens[256:])

        assert lp_split.shape[0] == 300
        width = lp_split.size(1)
        assert lp_a.size(1) == width  # longest sequence lives in chunk 1
        assert lp_b.size(1) < width  # chunk 2 is shorter -> was padded

        pad_cols = width - lp_b.size(1)
        lp_ref = torch.cat([lp_a, torch.nn.functional.pad(lp_b, (0, pad_cols))], 0)
        mask_ref = torch.cat([mask_a, torch.nn.functional.pad(mask_b, (0, pad_cols))], 0)
        assert torch.allclose(lp_split, lp_ref, atol=1e-7)
        assert torch.equal(mask_split, mask_ref)
        print("[OK] no-grad logprob chunking matches manual concatenation")
    finally:
        trainer.cleanup()


def test_grpo_chunked_loss_matches_single_forward(tmp_path):
    """Token-weighted chunk losses must reconstruct the single-forward loss."""
    trainer = _make_trainer(str(tmp_path))
    try:
        rollout, flat_seqs, flat_plens, flat_rlens = _build_rollout(trainer)
        rewards_t = torch.from_numpy(rollout["rewards"])
        advantages = trainer.compute_advantages(rewards_t, "group", 0.0)

        trainer.model.train()
        loss_flat, m_flat = trainer._grpo_loss_from_flat(
            rollout, advantages, rewards_t, flat_seqs, flat_plens, flat_rlens, G, B
        )

        # Replicate compute_grpo_loss's chunk loop without the backward.
        total_tokens = int(sum(rollout["response_lens"][g][b] for g in range(G) for b in range(B)))
        total_tokens = max(total_tokens, 1)
        prompt_chunk = max(1, 256 // G)  # 64
        loss_sum = torch.zeros(())
        w_policy = w_kl = w_tok = 0.0
        for start in range(0, B, prompt_chunk):
            end = min(B, start + prompt_chunk)
            seqs, plens, rlens = [], [], []
            for g in range(G):
                for b in range(start, end):
                    seqs.append(rollout["prompt_tokens"][b] + rollout["response_ids"][g][b])
                    plens.append(rollout["prompt_lens"][g][b])
                    rlens.append(rollout["response_lens"][g][b])
            loss_c, m_c, tokens_c = trainer._grpo_surrogate(
                rollout, advantages[:, start:end], seqs, plens, rlens, G, end - start, start
            )
            loss_sum = loss_sum + loss_c * (tokens_c / total_tokens)
            w_policy += m_c["policy_loss"] * tokens_c
            w_kl += m_c["kl_loss"] * tokens_c
            w_tok += tokens_c

        assert torch.allclose(
            loss_flat, loss_sum, atol=1e-5
        ), f"flat={loss_flat.item():.6f} chunked={loss_sum.item():.6f}"
        assert abs(m_flat["policy_loss"] - w_policy / w_tok) < 1e-5
        assert abs(m_flat["kl_loss"] - w_kl / w_tok) < 1e-6
        print(
            f"[OK] chunked GRPO loss matches single-forward "
            f"(loss={loss_flat.item():.6f}, chunks={(B + prompt_chunk - 1) // prompt_chunk})"
        )
    finally:
        trainer.cleanup()


def test_compute_grpo_loss_chunked_path_backward_smoke(tmp_path):
    """End-to-end chunked compute: detached return, internal backward, grads."""
    trainer = _make_trainer(str(tmp_path))
    try:
        rollout, _, _, _ = _build_rollout(trainer)
        trainer.optimizer.zero_grad(set_to_none=True)

        loss, metrics = trainer.compute_grpo_loss(rollout)

        # The chunked path backwards internally and returns a logging scalar.
        assert not loss.requires_grad
        for key in (
            "policy_loss",
            "kl_loss",
            "mean_reward",
            "std_reward",
            "mean_advantage",
            "std_advantage",
        ):
            assert key in metrics, f"missing metric {key}"
            assert np.isfinite(metrics[key]), f"non-finite {key}={metrics[key]}"
        grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
        assert grads, "chunked path must accumulate gradients"
        print(f"[OK] chunked compute_grpo_loss backward (policy_loss={metrics['policy_loss']:.4f})")
    finally:
        trainer.cleanup()
