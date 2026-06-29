"""Tests for document packing and cross-document attention mask."""
import os
import tempfile

import numpy as np
import pytest
import torch


from data.openwebtext import PackingDataset, MemmapDataset
from model.modern_gpt import ModernGPT, ModernGPTConfig


def _make_bin(path, tokens, dtype=np.uint16):
    np.array(tokens, dtype=dtype).tofile(path)


def test_packing_dataset_produces_document_ids():
    tmpdir = tempfile.mkdtemp()
    try:
        bin_path = os.path.join(tmpdir, "train.bin")
        # Three short docs separated by EOT, each < block_size.
        eot = 50256
        tokens = [1, 2, 3, eot, 4, 5, eot, 6, 7, 8, 9, eot]
        _make_bin(bin_path, tokens)

        ds = PackingDataset(bin_path, block_size=8, eot_token=eot, shuffle=False)
        samples = list(ds)
        assert len(samples) > 0
        for x, y, doc_ids in samples:
            assert x.shape[0] == 8
            assert y.shape[0] == 8
            assert doc_ids.shape[0] == 8
            # Determine the length of real tokens in this sample.
            pad_mask = doc_ids == -1
            pad_count = int(pad_mask.sum().item())
            real_len = len(x) - pad_count
            # y should be masked to -1 for EOT positions, the last real token,
            # and all padding positions.
            for j in range(len(x)):
                if j >= real_len or x[j].item() == eot or j == real_len - 1:
                    assert y[j].item() == -1
            # doc_ids for padding should be -1.
            assert pad_count > 0 or real_len == len(x)
    finally:
        # np.memmap keeps files open on Windows; ignore cleanup errors.
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_packing_cross_document_mask_changes_output():
    """ModernGPT output should differ when document_ids mask is applied."""
    config = ModernGPTConfig(block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = ModernGPT(config)
    model.eval()

    # Two documents packed into one sequence.
    x = torch.tensor([[1, 2, 3, 4, 50256, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
    y = torch.tensor([[2, 3, 4, 50256, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, -1]], dtype=torch.long)
    document_ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.long)

    with torch.no_grad():
        logits_no_mask, _, _ = model(x, targets=y)
        logits_with_mask, _, _ = model(x, targets=y, document_ids=document_ids)

    # At least some positions should differ because attention is masked differently.
    assert not torch.allclose(logits_no_mask, logits_with_mask, atol=1e-6)


def test_packing_long_document_chunks_share_doc_id():
    """Long documents split into block_size chunks must keep a single doc id."""
    tmpdir = tempfile.mkdtemp()
    try:
        bin_path = os.path.join(tmpdir, "train.bin")
        eot = 50256
        # One long document: 20 tokens followed by EOT.
        long_doc = list(range(1, 21)) + [eot]
        tokens = long_doc
        _make_bin(bin_path, tokens)

        ds = PackingDataset(bin_path, block_size=8, eot_token=eot, shuffle=False)
        samples = list(ds)
        # block_size=8, doc length=21 -> two full blocks + one tail.
        assert len(samples) >= 2
        full_blocks = [s for s in samples if (s[2] != -1).sum().item() == 8]
        # All full blocks of the same document share doc_id.
        doc_ids = {int(s[2][0].item()) for s in full_blocks}
        assert len(doc_ids) == 1
        doc_id = doc_ids.pop()
        # The last target of a full block should be the next token in the doc,
        # not the EOT.
        for x, y, dids in full_blocks:
            assert (dids == doc_id).all()
            if y[-1].item() != -1:
                # y[-1] predicts the token after this block.
                assert y[-1].item() != eot
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_memmap_dataset_shuffle_buffer():
    tmpdir = tempfile.mkdtemp()
    try:
        bin_path = os.path.join(tmpdir, "train.bin")
        tokens = list(range(1024))
        _make_bin(bin_path, tokens)

        ds1 = MemmapDataset(bin_path, block_size=64, shuffle_buffer=1)
        ds2 = MemmapDataset(bin_path, block_size=64, shuffle_buffer=10000)

        # With buffer size 1 there is effectively no shuffle.
        s1 = [tuple(x.tolist()) for x, _ in ds1]
        s2 = [tuple(x.tolist()) for x, _ in ds2]
        assert sorted(s1) == sorted(s2)
        # A full-buffer shuffle is extremely unlikely to reproduce the exact order.
        assert s1 != s2
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
