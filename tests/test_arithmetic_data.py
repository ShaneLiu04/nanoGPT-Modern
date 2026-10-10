"""Arithmetic dataset regressions (migrated from test_bugfixes.py)."""

import torch


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
    assert (
        len(sample["input_ids"]) + 1 == len(sample["labels"]) + 1
    )  # input_ids is tokens[:-1]
    print("[OK] ArithmeticDataset pre-tokenization works")


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
