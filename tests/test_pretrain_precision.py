"""Regression locks for the ``--precision`` flag (E-C-1 cloud ablation).

``fp32`` must short-circuit AMP entirely (nullcontext, no GradScaler,
``amp_dtype`` float32) -- a silent fallthrough to autocast would
contaminate precision ablations. ``fp16`` must route through the
mixed-precision path with an enabled GradScaler.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from training.train_pretrain import PretrainTrainer


class _AmpProbe(PretrainTrainer):
    """Trainer shell exposing exactly what ``_setup_amp`` touches.

    Subclassing keeps the zero-argument ``super()`` inside
    ``PretrainTrainer._setup_amp`` valid; ``__init__`` is overridden so the
    heavy BaseTrainer construction never runs.
    """

    def __init__(self, precision, device):
        self.args = SimpleNamespace(use_ema=False, precision=precision)
        self.raw_model = None
        self.ctx = None
        self.scaler = None
        self.amp_dtype = None
        self.device = device


def test_precision_fp32_disables_amp():
    probe = _AmpProbe("fp32", "cpu")
    PretrainTrainer._setup_amp(probe, use_bf16=True)
    assert isinstance(probe.ctx, nullcontext)
    assert probe.scaler is None
    assert probe.amp_dtype is torch.float32
    print("[OK] --precision fp32 disables autocast and GradScaler")


def test_precision_default_still_routes_through_amp_setup():
    probe = _AmpProbe("bf16", "cpu")
    PretrainTrainer._setup_amp(probe, use_bf16=True)
    # CPU baseline: nullcontext + float32, but via the AMP setup path.
    assert probe.amp_dtype is torch.float32
    assert probe.scaler is None
    print("[OK] default precision routes through the AMP setup path")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 needs CUDA")
def test_precision_fp16_enables_grad_scaler_on_cuda():
    probe = _AmpProbe("fp16", "cuda")
    PretrainTrainer._setup_amp(probe, use_bf16=True)
    assert probe.amp_dtype is torch.float16
    assert probe.scaler is not None and probe.scaler.is_enabled()
    print("[OK] --precision fp16 selects fp16 autocast + GradScaler")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp32-vs-amp needs CUDA")
def test_precision_fp32_short_circuits_on_cuda():
    probe = _AmpProbe("fp32", "cuda")
    PretrainTrainer._setup_amp(probe, use_bf16=True)
    assert isinstance(probe.ctx, nullcontext)
    assert probe.amp_dtype is torch.float32
    assert probe.scaler is None
    print("[OK] --precision fp32 short-circuits AMP on CUDA")
