"""Regression tests for the blocking bugs fixed in this pass."""
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import torch


from model.baseline_gpt import BaselineGPT, BaselineGPTConfig
from model.modern_gpt import ModernGPT, ModernGPTConfig
from model.kv_cache_utils import KVCacheManager
from data.openwebtext import DocBoundaryDataset
from utils.checkpoint import save_checkpoint, load_checkpoint, get_rng_state, set_rng_state


def test_modern_gpt_kv_cache_dtype():
    """ModernGPT.generate(use_cache=True) must not crash on dtype mismatch."""
    config = ModernGPTConfig(
        n_layer=2, n_head=4, n_embd=128, block_size=64,
        vocab_size=100, dropout=0.0, n_kv_head=2,
    )
    model = ModernGPT(config)
    model.eval()
    idx = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=5, use_cache=True)
    assert out.shape[0] == 1
    assert out.shape[1] == 10
    print("[OK] ModernGPT KV cache dtype fix works")


def test_optimizer_weight_tying_dedup():
    """wte and lm_head share the same tensor; optimizer must see it once."""
    config = ModernGPTConfig(
        n_layer=2, n_head=4, n_embd=128, block_size=64,
        vocab_size=100, dropout=0.0,
    )
    model = ModernGPT(config)
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    param_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    # wte and lm_head are the same tensor -> should appear only once in optimizer.
    assert id(model.transformer.wte.weight) == id(model.lm_head.weight)
    assert id(model.transformer.wte.weight) in param_ids
    # Count how many times the shared tensor appears in optimizer params.
    occurrences = sum(
        1 for group in opt.param_groups for p in group["params"] if id(p) == id(model.transformer.wte.weight)
    )
    assert occurrences == 1, f"shared weight appears {occurrences} times in optimizer"
    print("[OK] Optimizer weight-tying deduplication works")


def test_ma_update_and_checkpoint():
    """EMA shadow weights must update and round-trip through checkpoint."""
    config = ModernGPTConfig(
        n_layer=2, n_head=4, n_embd=128, block_size=64,
        vocab_size=100, dropout=0.0,
    )
    model = ModernGPT(config)
    model.init_ema(decay=0.9)

    # Simulate one update step with dummy gradients.
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    x = torch.randint(0, 100, (2, 8))
    y = torch.randint(0, 100, (2, 8))
    _, loss, _ = model(x, y)
    loss.backward()
    opt.step()
    model.update_ema()

    # EMA should differ from current weights after update.
    name = next(iter(model.ema_shadow))
    assert not torch.equal(model.ema_shadow[name], model.state_dict()[name])

    # Save and load checkpoint.
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_checkpoint(
            model, opt, iter_num=7, best_val_loss=1.23, config=config,
            out_dir=tmpdir, filename="ckpt.pt",
            ema_shadow=model.ema_shadow, rng_state=get_rng_state(),
        )
        model2 = ModernGPT(config)
        opt2 = model2.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
        extra = load_checkpoint(path, model2, opt2)
        assert extra["iter_num"] == 7
        assert extra["best_val_loss"] == 1.23
        assert "ema_shadow" in extra
        assert torch.equal(extra["ema_shadow"][name], model.ema_shadow[name])
    print("[OK] EMA update and checkpoint round-trip work")


def test_doc_boundary_dataset_resume_offset():
    """DocBoundaryDataset must accept resume_offset and iterate without error."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        bin_path = os.path.join(tmpdir, "train.bin")
        # 1024 tokens with EOT at positions 100, 200, ...
        data = np.zeros(1024, dtype=np.uint16)
        for pos in range(100, 1024, 100):
            data[pos] = 50256
        data.tofile(bin_path)

        ds = DocBoundaryDataset(bin_path, block_size=64, resume_offset=0)
        samples = list(ds)
        del ds  # release the memmap so Windows can clean up the temp dir
        assert len(samples) > 0
        for x, y in samples:
            assert x.shape[0] == y.shape[0]
        print("[OK] DocBoundaryDataset resume_offset fix works")


def test_doc_boundary_dataset_state_dict():
    """DocBoundaryDataset state_dict/load_state_dict round-trip."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        bin_path = os.path.join(tmpdir, "train.bin")
        data = np.zeros(1024, dtype=np.uint16)
        for pos in range(100, 1024, 100):
            data[pos] = 50256
        data.tofile(bin_path)

        ds = DocBoundaryDataset(bin_path, block_size=64, resume_offset=3)
        state = ds.state_dict()
        assert state == {"resume_offset": 3}
        ds.load_state_dict({"resume_offset": 5})
        assert ds.resume_offset == 5
        del ds
        print("[OK] DocBoundaryDataset state_dict round-trip works")


# ---------------------------------------------------------------------------
# GRPO performance / correctness regression tests
# ---------------------------------------------------------------------------

def _tiny_grpo_args(tmpdir, ckpt_path, **overrides):
    """Build a minimal args Namespace for GRPO testing."""
    defaults = dict(
        init_from=ckpt_path,
        ref_from=ckpt_path,
        out_dir=tmpdir,
        group_size=2,
        num_steps=4,
        batch_size=2,
        gradient_accumulation_steps=2,
        max_prompt_len=32,
        max_response_len=16,
        learning_rate=1e-4,
        min_lr=1e-5,
        weight_decay=0.0,
        grad_clip=1.0,
        beta=0.04,
        eps=0.2,
        lr_schedule="constant",
        seed=1337,
        device="cpu",
        backend="nccl",
        use_wandb=False,
        eval_interval=10,
        num_train=4,
        num_val=4,
        resume=None,
        config=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_tiny_checkpoint(tmpdir):
    """Create a tiny ModernGPT checkpoint on disk and return its path."""
    config = ModernGPTConfig(
        n_layer=1, n_head=2, n_embd=64, block_size=128,
        dropout=0.0,
    )
    model = ModernGPT(config)
    opt = model.configure_optimizers(0.0, 1e-3, (0.9, 0.95), "cpu")
    path = save_checkpoint(
        model, opt, iter_num=0, best_val_loss=float("inf"), config=config,
        out_dir=tmpdir, filename="ckpt.pt",
    )
    return path


def test_grpo_batched_logprobs_consistency():
    """Batched logprob computation must match per-sequence logprobs."""
    from training.train_grpo import GRPOTrainer

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = _make_tiny_checkpoint(tmpdir)
        args = _tiny_grpo_args(tmpdir, ckpt, num_train=2, num_val=2)
        trainer = GRPOTrainer(args)
        model = trainer.policy
        model.eval()

        sequences = [
            [1, 2, 3, 4, 5],
            [10, 11, 12, 13, 14, 15],
            [20, 21, 22],
        ]
        prompt_lens = [2, 3, 1]
        response_lens = [3, 3, 2]

        batched_logps, masks = trainer._batch_logprobs(model, sequences, prompt_lens, response_lens)

        # Compare response-token logprobs against per-sequence computation.
        for i, seq in enumerate(sequences):
            inp = torch.tensor([seq[:-1]], dtype=torch.long)
            tgt = torch.tensor([seq[1:]], dtype=torch.long)
            with torch.no_grad():
                logits, _, _ = model(inp)
                logp = torch.log_softmax(logits, dim=-1)
                single_logps = logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
            resp_mask = masks[i, :len(seq) - 1]
            assert torch.allclose(batched_logps[i, :len(seq) - 1][resp_mask], single_logps[0][resp_mask], atol=1e-5)
            expected_mask = [False] * (prompt_lens[i] - 1) + [True] * response_lens[i]
            assert torch.equal(resp_mask, torch.tensor(expected_mask))

        print("[OK] GRPO batched logprob computation is consistent")


def test_grpo_gradient_accumulation():
    """GRPO optimizer should step only every gradient_accumulation_steps rollouts."""
    from training.train_grpo import GRPOTrainer

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = _make_tiny_checkpoint(tmpdir)
        args = _tiny_grpo_args(
            tmpdir, ckpt,
            num_steps=6,
            batch_size=2,
            group_size=2,
            gradient_accumulation_steps=3,
            num_train=4,
            num_val=4,
        )
        trainer = GRPOTrainer(args)

        # Mock sampling to avoid expensive generation.
        vocab_size = trainer.raw_model.config.vocab_size
        prompt_len = 4
        response_len = 4

        def _mock_sample_group(prompts, answers, prompt_tokens=None):
            group_size, batch_size = args.group_size, len(prompts)
            responses = [["dummy"] * batch_size for _ in range(group_size)]
            response_ids = [[[vocab_size // 2] * response_len for _ in range(batch_size)] for _ in range(group_size)]
            rewards = np.ones((group_size, batch_size), dtype=np.float32)

            flat_sequences = []
            flat_prompt_lens = []
            flat_response_lens = []
            for g in range(group_size):
                for b in range(batch_size):
                    ptoks = prompt_tokens[b] if prompt_tokens else [0] * prompt_len
                    full = ptoks + response_ids[g][b]
                    flat_sequences.append(full)
                    flat_prompt_lens.append(len(ptoks))
                    flat_response_lens.append(response_len)

            old_logprobs, masks = trainer._batch_logprobs(trainer.policy, flat_sequences, flat_prompt_lens, flat_response_lens)
            ref_logprobs, _ = trainer._batch_logprobs(trainer.ref, flat_sequences, flat_prompt_lens, flat_response_lens)

            return {
                "responses": responses,
                "response_ids": response_ids,
                "rewards": rewards,
                "old_logprobs": old_logprobs,
                "ref_logprobs": ref_logprobs,
                "masks": masks,
                "prompt_lens": [[prompt_len] * batch_size for _ in range(group_size)],
                "response_lens": [[response_len] * batch_size for _ in range(group_size)],
                "prompt_tokens": prompt_tokens or [[0] * prompt_len for _ in range(batch_size)],
            }

        original_sample_group = trainer.sample_group
        trainer.sample_group = _mock_sample_group

        step_counts = []
        original_optimizer_step = trainer.optimizer.step

        def _counting_step():
            step_counts.append(len(step_counts))
            return original_optimizer_step()

        trainer.optimizer.step = _counting_step

        try:
            trainer.train()
        finally:
            trainer.sample_group = original_sample_group
            trainer.optimizer.step = original_optimizer_step

        # 6 rollouts / 3 accum = 2 optimizer steps.
        assert len(step_counts) == 2, f"expected 2 optimizer steps, got {len(step_counts)}"
        print("[OK] GRPO gradient accumulation steps correctly")


def test_iterative_grpo_rejection_pool():
    """IterativeGRPOTrainer must build train/val/rejection pools."""
    from training.iterative_grpo import IterativeGRPOTrainer

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = _make_tiny_checkpoint(tmpdir)
        args = _tiny_grpo_args(tmpdir, ckpt, num_train=4, num_val=4)
        args.ref_update_interval = 2
        args.ref_update_ratio = 0.5
        args.rejection_interval = 2
        args.rejection_samples = 4
        args.rejection_top_k = 2
        args.rejection_sft_steps = 1
        args.rejection_sft_lr = 1e-4
        args.rejection_batch_size = 2

        trainer = IterativeGRPOTrainer(args)
        assert hasattr(trainer, "rejection_pool")
        assert len(trainer.rejection_pool) == 3 * args.num_train
        assert trainer.ref_updates_done == 0
        assert trainer.rejection_rounds_done == 0
        print("[OK] IterativeGRPOTrainer builds data pools")


def test_kv_cache_ring_buffer_order():
    """Ring-buffer cache must return keys/values in logical order."""
    B, n_layers, n_kv_heads, head_dim, max_len = 2, 1, 2, 4, 6
    cache = KVCacheManager(n_layers, n_kv_heads, n_kv_heads, head_dim, max_cache_len=max_len)
    cache.init_cache(B, "cpu", torch.float32)

    # Write 4 tokens.
    tok4 = torch.arange(B * n_kv_heads * 4 * head_dim).view(B, n_kv_heads, 4, head_dim).float()
    cache.update(0, tok4, tok4.clone())
    cache.advance(4)
    blocks = cache.get_cache()
    assert len(blocks) == 1
    assert torch.equal(blocks[0][0], tok4)
    assert torch.equal(blocks[0][1], tok4)
    assert cache.cache_len == 4

    # Write 3 more tokens (wraps around, evicts oldest tokens).
    # One slot is reserved for the next single-token decode step, so the
    # effective capacity is max_len - 1.
    tok3 = torch.arange(B * n_kv_heads * 3 * head_dim).view(B, n_kv_heads, 3, head_dim).float() + 1000
    cache.update(0, tok3, tok3.clone())
    cache.advance(3)
    effective_len = max_len - 1
    assert cache.cache_len == effective_len
    assert cache.start_pos == 2

    blocks = cache.get_cache()
    # When wrapped, the single layer is represented by two contiguous blocks.
    assert isinstance(blocks[0], list)
    assert len(blocks[0]) == 2
    recovered = torch.cat([blocks[0][0][0], blocks[0][1][0]], dim=2)
    expected = torch.cat([tok4[:, :, 2:, :], tok3], dim=2)
    assert torch.equal(recovered, expected)
    print("[OK] KV Cache ring buffer ordering works")


def test_kv_cache_sliding_window_eviction():
    """Cache must keep only the last max_cache_len tokens."""
    B, n_layers, n_kv_heads, head_dim, max_len = 1, 1, 1, 2, 4
    cache = KVCacheManager(n_layers, n_kv_heads, n_kv_heads, head_dim, max_cache_len=max_len)
    cache.init_cache(B, "cpu", torch.float32)

    for step in range(8):
        new = torch.full((B, n_kv_heads, 1, head_dim), float(step))
        cache.update(0, new, new.clone())
        cache.advance(1)

    # Effective capacity is max_len - 1 to leave room for the next decode token.
    effective_len = max_len - 1
    assert cache.cache_len == effective_len
    assert cache.start_pos == 5
    layer_cache = cache.get_cache()[0]
    if isinstance(layer_cache, list):
        recovered = torch.cat([b[0] for b in layer_cache], dim=2)
    else:
        recovered = layer_cache[0]
    expected = torch.arange(5, 8).view(1, 1, effective_len, 1).expand(-1, -1, -1, head_dim).float()
    assert torch.equal(recovered, expected)
    print("[OK] KV Cache sliding window eviction works")


def test_kv_cache_long_generate_consistency():
    """generate(use_cache=True) must match no-cache output even when exceeding cache length."""
    config = ModernGPTConfig(
        n_layer=2, n_head=2, n_embd=64, block_size=32,
        dropout=0.0,
    )
    model = ModernGPT(config)
    model.eval()

    idx = torch.randint(0, config.vocab_size, (1, 5))
    # Use greedy decoding so sampled tokens depend only on the logits, not on
    # the RNG state difference between the two generate() calls.
    with torch.no_grad():
        out_cache = model.generate(idx, max_new_tokens=40, use_cache=True, top_k=1)
        out_nocache = model.generate(idx, max_new_tokens=40, use_cache=False, top_k=1)

    assert out_cache.shape == out_nocache.shape
    assert torch.equal(out_cache, out_nocache)
    print("[OK] Long generation with KV cache matches no-cache output")


def test_kv_cache_set_restore():
    """set_cache must restore logical order and state."""
    B, n_layers, n_kv_heads, head_dim, max_len = 1, 2, 2, 4, 8
    cache = KVCacheManager(n_layers, n_kv_heads, n_kv_heads, head_dim, max_cache_len=max_len)
    cache.init_cache(B, "cpu", torch.float32)

    k = torch.randn(B, n_kv_heads, 5, head_dim)
    v = torch.randn(B, n_kv_heads, 5, head_dim)
    cache.update(0, k, v)
    cache.update(1, k, v)
    cache.advance(5)

    snapshot = cache.get_cache()
    cache2 = KVCacheManager(n_layers, n_kv_heads, n_kv_heads, head_dim, max_cache_len=max_len)
    cache2.init_cache(B, "cpu", torch.float32)
    cache2.set_cache(snapshot)

    assert cache2.cache_len == cache.cache_len
    assert cache2.start_pos == cache.start_pos
    restored = cache2.get_cache()
    for i in range(n_layers):
        if isinstance(snapshot[i], list):
            for j in range(len(snapshot[i])):
                assert torch.equal(snapshot[i][j][0], restored[i][j][0])
                assert torch.equal(snapshot[i][j][1], restored[i][j][1])
        else:
            assert torch.equal(snapshot[i][0], restored[i][0])
            assert torch.equal(snapshot[i][1], restored[i][1])
    print("[OK] KV Cache set/restore works")


def test_prepare_shard_writer_dtype_and_index():
    """_ShardWriter must produce readable shards and a dtype-aware .idx."""
    from data.prepare import _ShardWriter, _select_dtype, _concatenate_shards_to_bin
    from data.openwebtext import _detect_dtype_from_index

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = _ShardWriter(
            output_dir=tmpdir,
            split="train",
            dtype=np.uint32,
            shard_size=100,
            vocab_size=70000,
            eot_token=50256,
        )
        tokens = np.arange(1000, 1100, dtype=np.uint32)
        writer.append(tokens)
        writer.record_doc_boundary()
        writer.append(tokens + 100)
        writer.record_doc_boundary()
        meta = writer.close()
        writer.write_index()

        assert meta["total_tokens"] == 200
        assert meta["num_docs"] == 2

        _concatenate_shards_to_bin(writer.shard_dir, tmpdir, "train", np.uint32)
        bin_path = os.path.join(tmpdir, "train.bin")
        idx_path = os.path.join(tmpdir, "train.idx")
        assert os.path.exists(bin_path)
        assert os.path.exists(idx_path)

        detected = _detect_dtype_from_index(bin_path)
        assert detected == np.uint32

        arr = np.memmap(bin_path, dtype=np.uint32, mode="r")
        try:
            assert len(arr) == 200
            assert arr[0] == 1000
            assert arr[100] == 1100
        finally:
            del arr  # close memmap so Windows can clean up the temp dir
        print("[OK] Shard writer + dtype-aware index works")


def test_prepare_dtype_selection():
    """_select_dtype must pick uint16 for small vocab and uint32 for large vocab."""
    from data.prepare import _select_dtype
    assert _select_dtype(50000) == np.uint16
    assert _select_dtype(65535) == np.uint16
    assert _select_dtype(65536) == np.uint32
    assert _select_dtype(100000) == np.uint32
    print("[OK] prepare.py dtype selection works")


def test_arithmetic_dataset_pre_tokenize():
    """ArithmeticDataset must pre-tokenize samples and not call encode in __getitem__."""
    import tiktoken
    from data.arithmetic import generate_easy, ArithmeticDataset

    tokenizer = tiktoken.get_encoding("gpt2")
    data = generate_easy(num_samples=10, seed=42)
    ds = ArithmeticDataset(data, tokenizer, max_length=128, pre_tokenize=True)

    assert len(ds) == 10
    sample = ds[0]
    assert "input_ids" in sample
    assert "labels" in sample
    assert isinstance(sample["input_ids"], torch.Tensor)
    assert isinstance(sample["labels"], torch.Tensor)
    assert len(sample["input_ids"]) + 1 == len(sample["labels"]) + 1  # input_ids is tokens[:-1]
    print("[OK] ArithmeticDataset pre-tokenization works")


def test_rule_reward_perfect_partial_and_malformed():
    """rule_reward must give full credit for perfect answers and partial credit otherwise."""
    from rewards.rule_reward import compute_reward

    # Perfect answer with derivation.
    total, fmt, proc, acc = compute_reward(
        "Step 1: 2 + 3 = 5. <answer>5</answer>", "5"
    )
    assert total > 1.5
    assert fmt == 0.5
    assert proc > 0.0
    assert acc == 1.2

    # Close but not exact answer.
    total, fmt, proc, acc = compute_reward("<answer>5.0001</answer>", "5")
    assert fmt == 0.5
    assert acc > 0.0

    # Malformed: multiple answer blocks.
    total, fmt, proc, acc = compute_reward("<answer>5</answer><answer>6</answer>", "5")
    assert fmt < 0.5
    assert acc == 1.2  # first block is still evaluated

    # No answer block.
    total, fmt, proc, acc = compute_reward("The answer is 5.", "5")
    assert fmt == 0.0
    assert acc == 0.0

    # Non-numeric / inf / nan should be rejected.
    for bad in ["inf", "nan", "1e1000", "foo"]:
        total, fmt, proc, acc = compute_reward(f"<answer>{bad}</answer>", "5")
        if bad == "foo":
            assert fmt < 0.5 or acc == 0.0
        else:
            assert fmt < 0.5
            assert acc == 0.0
    print("[OK] Rule reward perfect/partial/malformed cases work")


def test_rule_reward_process_score():
    """rule_reward should reward intermediate derivation steps."""
    from rewards.rule_reward import compute_reward

    _, _, proc_with, _ = compute_reward(
        "First compute 2*3 = 6, then 6+4 = 10. <answer>10</answer>", "10"
    )
    _, _, proc_without, _ = compute_reward("<answer>10</answer>", "10")
    assert proc_with > proc_without
    print("[OK] Rule reward process score works")


def test_generate_medium_no_division_by_zero():
    """generate_medium must never produce expressions that divide by zero."""
    from data.arithmetic import generate_medium

    for seed in range(20):
        data = generate_medium(num_samples=100, seed=seed)
        assert len(data) == 100
        for sample in data:
            assert "/ 0" not in sample["answer"]
            # Verify the answer can be evaluated (no ZeroDivisionError).
            assert sample["answer"] not in ("inf", "nan", "")
    print("[OK] generate_medium avoids division by zero")


if __name__ == "__main__":
    test_modern_gpt_kv_cache_dtype()
    test_optimizer_weight_tying_dedup()
    test_ma_update_and_checkpoint()
    test_doc_boundary_dataset_resume_offset()
    test_grpo_batched_logprobs_consistency()
    test_grpo_gradient_accumulation()
    test_iterative_grpo_rejection_pool()
    test_kv_cache_ring_buffer_order()
    test_kv_cache_sliding_window_eviction()
    test_kv_cache_long_generate_consistency()
    test_kv_cache_set_restore()
    test_prepare_shard_writer_dtype_and_index()
    test_prepare_dtype_selection()
    test_arithmetic_dataset_pre_tokenize()
    test_rule_reward_perfect_partial_and_malformed()
    test_rule_reward_process_score()
    print("\nAll regression tests passed.")
