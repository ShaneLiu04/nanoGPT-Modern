"""Regression locks for the DPO jsonl preference source and the
response-mask logprob alignment fix.

The cloud DPO/IPO comparison runs introduced:

* ``JsonlPreferenceDataset`` + ``collate_preference``: preference pairs as
  token ids plus a response mask, right-padded to the batch width with
  mask 0 on padding.
* a realignment of ``DPOTrainer._compute_logprobs``: ``logits[:, t]`` must
  score ``tokens[:, t + 1]`` (the previous code scored ``tokens[:, t]``,
  training on the wrong token), with the mean restricted to response
  positions via the mask.
"""

import argparse
import json
import os
import shutil
import tempfile

import numpy as np
import pytest
import torch
from test_dpo_trainer import _make_minimal_checkpoint

from training.train_dpo import DPOTrainer, JsonlPreferenceDataset, collate_preference


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _dpo_args(ckpt, out_dir, **overrides):
    defaults = dict(
        init_from=ckpt,
        ref_from=None,
        out_dir=out_dir,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_length=16,
        epochs=1,
        learning_rate=1e-4,
        min_lr=1e-5,
        lr_schedule="cosine",
        warmup_iters=0,
        weight_decay=0.0,
        grad_clip=1.0,
        beta=0.1,
        preference_loss="dpo",
        preference_source="synthetic",
        preference_jsonl=None,
        preference_val_jsonl=None,
        label_smoothing=0.0,
        eval_interval=1000,
        seed=42,
        device="cpu",
        use_wandb=False,
        backend="gloo",
        keep_last_n=0,
        resume=None,
        config=None,
        attn_backend="auto",
        num_train=8,
        num_val=4,
        vocab_size=50257,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_jsonl_dataset_and_collate_roundtrip(tmp_path):
    rows = [
        dict(
            chosen=[1, 2, 3, 4],
            rejected=[1, 2, 9, 4, 5],
            chosen_mask=[0, 0, 1, 1],
            rejected_mask=[0, 0, 1, 1, 1],
        ),
        dict(
            chosen=[5, 6],
            rejected=[7, 8],
            chosen_mask=[0, 1],
            rejected_mask=[0, 1],
        ),
    ]
    path = tmp_path / "pref.jsonl"
    _write_jsonl(str(path), rows)

    ds = JsonlPreferenceDataset(str(path))
    assert len(ds) == 2
    assert ds[0][0] == [1, 2, 3, 4]
    assert ds[0][3] == [0, 0, 1, 1, 1]

    chosen, rejected, chosen_mask, rejected_mask = collate_preference([ds[0], ds[1]])
    assert chosen.shape == (2, 4)
    assert rejected.shape == (2, 5)
    assert chosen_mask[0].tolist() == [0, 0, 1, 1]
    # Right padding stays at mask 0.
    assert chosen_mask[1].tolist() == [0, 1, 0, 0]
    assert rejected_mask[1].tolist() == [0, 1, 0, 0, 0]

    # Batch width never drops below 2 (single short row).
    lone = collate_preference([([9], [8], [0], [0])])
    assert lone[0].shape == (1, 2)

    # Empty file is rejected.
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        JsonlPreferenceDataset(str(empty))
    print("[OK] jsonl preference dataset + collate round-trip")


def test_dpo_compute_logprobs_alignment_and_mask(tmp_path):
    """logits[:, t] scores tokens[:, t + 1]; mask restricts the mean."""
    tmpdir = tempfile.mkdtemp(dir=str(tmp_path))
    try:
        ckpt = os.path.join(tmpdir, "sft.pt")
        _make_minimal_checkpoint(ckpt)
        trainer = DPOTrainer(_dpo_args(ckpt, os.path.join(tmpdir, "out")))

        torch.manual_seed(5)
        tokens = torch.tensor([[10, 11, 12, 13, 14]])
        # Tokens 12..14 are the response (mask marks response positions).
        mask = torch.tensor([[0, 0, 1, 1, 1]])

        lp = trainer._compute_logprobs(trainer.model, tokens, mask)

        with torch.no_grad():
            logits, _, _ = trainer.model(tokens[:, :-1])
            logp = torch.log_softmax(logits, dim=-1)
            tok_lp = logp.gather(2, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
            m = mask[:, 1:].bool()
            expected = (tok_lp * m).sum() / m.sum()

        assert torch.allclose(
            lp, expected, atol=1e-6
        ), f"aligned={lp.item():.6f} expected={expected.item():.6f}"

        # Mask position 0 never contributes (no prediction for token 0).
        lp_flip = trainer._compute_logprobs(trainer.model, tokens, torch.tensor([[1, 0, 1, 1, 1]]))
        assert torch.allclose(lp, lp_flip, atol=1e-7)

        # mask=None averages over every predicted position.
        lp_nomask = trainer._compute_logprobs(trainer.model, tokens, None)
        assert torch.allclose(lp_nomask, tok_lp.mean(), atol=1e-6)
        print(f"[OK] DPO logprob alignment locked (masked={lp.item():.6f})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_dpo_jsonl_end_to_end_train_step(tmp_path):
    """preference_source=jsonl plumbs the collate into both loaders."""
    tmpdir = tempfile.mkdtemp(dir=str(tmp_path))
    try:
        rows = [
            dict(
                chosen=[1, 2, 3, 4, 5],
                rejected=[1, 2, 9, 4],
                chosen_mask=[0, 1, 1, 1, 1],
                rejected_mask=[0, 1, 1, 1],
            ),
            dict(
                chosen=[1, 6, 7],
                rejected=[1, 6, 8, 9],
                chosen_mask=[0, 1, 1],
                rejected_mask=[0, 1, 1, 1],
            ),
            dict(
                chosen=[3, 4, 5, 6],
                rejected=[3, 4, 5, 9],
                chosen_mask=[0, 1, 1, 1],
                rejected_mask=[0, 1, 1, 1],
            ),
        ]
        pref = os.path.join(tmpdir, "pref_train.jsonl")
        _write_jsonl(pref, rows)

        ckpt = os.path.join(tmpdir, "sft.pt")
        _make_minimal_checkpoint(ckpt)
        args = _dpo_args(
            ckpt,
            os.path.join(tmpdir, "out"),
            preference_source="jsonl",
            preference_jsonl=pref,
        )
        trainer = DPOTrainer(args)

        batch = next(iter(trainer.train_loader))
        assert len(batch) == 4, "jsonl loader must yield (chosen, rejected, cmask, rmask)"

        loss = trainer._run_epoch(0, is_train=True)
        assert np.isfinite(loss) and loss >= 0
        print(f"[OK] DPO jsonl end-to-end train step (loss={loss:.4f})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
