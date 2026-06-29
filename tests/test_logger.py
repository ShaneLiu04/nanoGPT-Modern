"""Regression tests for utils.logging."""
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch


from utils.logging import Logger


def test_logger_degrades_on_tensorboard_failure():
    with mock.patch("torch.utils.tensorboard.SummaryWriter", side_effect=ImportError("no tb")):
        logger = Logger(
            project_name="test",
            run_name="tb_fail",
            log_dir=tempfile.mkdtemp(),
            use_wandb=False,
            use_tensorboard=True,
        )
        assert logger.writer is None
        logger.close()


def test_logger_degrades_on_wandb_failure():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(
            project_name="test",
            run_name="wandb_fail",
            log_dir=tmpdir,
            use_wandb=True,
            use_tensorboard=False,
        )
        assert logger.run is None
        logger.close()


def test_log_scalar_and_scalars(tmp_path):
    from torch.utils.tensorboard import SummaryWriter

    logger = Logger(
        project_name="test",
        run_name="scalar",
        log_dir=str(tmp_path),
        use_wandb=False,
        use_tensorboard=True,
    )
    assert logger.writer is not None
    logger.log_scalar("x", 1.0, step=0)
    logger.log_scalars({"y": 2.0, "z": 3.0}, step=1)
    logger.close()
    # SummaryWriter writes asynchronously; just ensure no exception.


def test_log_text_sample(tmp_path):
    logger = Logger(
        project_name="test",
        run_name="text",
        log_dir=str(tmp_path),
        use_wandb=False,
        use_tensorboard=True,
    )
    logger.log_text("sample", "hello world", step=0)
    logger.close()


def test_log_histogram(tmp_path):
    logger = Logger(
        project_name="test",
        run_name="hist",
        log_dir=str(tmp_path),
        use_wandb=False,
        use_tensorboard=True,
    )
    logger.log_histogram(" grads", torch.randn(100), step=0)
    logger.close()


def test_log_grad_norms(tmp_path):
    logger = Logger(
        project_name="test",
        run_name="grads",
        log_dir=str(tmp_path),
        use_wandb=False,
        use_tensorboard=True,
    )
    model = torch.nn.Linear(10, 1)
    loss = model(torch.randn(4, 10)).sum()
    loss.backward()
    logger.log_grad_norms(model, step=0)
    logger.close()


def test_log_memory_stats(tmp_path):
    logger = Logger(
        project_name="test",
        run_name="memory",
        log_dir=str(tmp_path),
        use_wandb=False,
        use_tensorboard=True,
    )
    # Should not raise even on CPU-only test runners.
    logger.log_memory_stats(step=0)
    logger.close()


def test_logger_close_is_idempotent(tmp_path):
    logger = Logger(
        project_name="test",
        run_name="close",
        log_dir=str(tmp_path),
        use_wandb=False,
        use_tensorboard=True,
    )
    logger.close()
    logger.close()
    assert logger.writer is None
