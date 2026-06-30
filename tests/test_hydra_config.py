"""Tests for the Hydra/OmegaConf configuration migration (M23)."""
import argparse
import os
import sys

import pytest


hydra = pytest.importorskip("hydra")
pytest.importorskip("omegaconf")

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import ListConfig, OmegaConf

from utils.config import NestedNamespace, parse_args_with_config, to_dict
from utils.hydra_utils import to_namespace, validate_required


def _config_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "hydra"))


def _clear_hydra():
    """Reset Hydra global state between composition calls in tests."""
    GlobalHydra.instance().clear()


class TestHydraUtils:
    def test_to_namespace_resolves_interpolation(self):
        cfg = OmegaConf.create({"a": 1, "b": {"c": 2, "d": "${a}"}, "e": [1, 2, 3]})
        ns = to_namespace(cfg)
        assert isinstance(ns, NestedNamespace)
        assert ns.a == 1
        assert ns.b.c == 2
        assert ns.b.d == 1
        assert ns.e == [1, 2, 3]
        assert ns.get("b.d") == 1

    def test_to_namespace_compatible_with_vars(self):
        cfg = OmegaConf.create({"x": 10, "y": {"z": 20}})
        ns = to_namespace(cfg)
        d = vars(ns)
        assert d["x"] == 10
        assert isinstance(d["y"], NestedNamespace)

    def test_to_dict_handles_dictconfig(self):
        cfg = OmegaConf.create({"a": 1, "b": {"c": "${a}"}})
        d = to_dict(cfg)
        assert d == {"a": 1, "b": {"c": 1}}

    def test_validate_required_missing(self):
        cfg = OmegaConf.create({"x": "???", "y": 1})
        with pytest.raises(ValueError, match="Missing required config keys"):
            validate_required(cfg)

    def test_validate_required_explicit_set(self):
        cfg = OmegaConf.create({"x": "???", "y": 1})
        validate_required(cfg, required={"y"})  # should not raise
        with pytest.raises(ValueError, match="Missing required config keys"):
            validate_required(cfg, required={"x"})


class TestHydraConfigs:
    def test_pretrain_config_loads_with_defaults(self):
        _clear_hydra()
        with initialize_config_dir(version_base=None, config_dir=_config_dir()):
            cfg = compose(config_name="pretrain")
            assert cfg.model == "modern"
            assert cfg.block_size == 1024
            assert cfg.n_layer == 12
            assert cfg.batch_size == 12
            assert cfg.learning_rate == 6.0e-4

    def test_pretrain_cli_override(self):
        _clear_hydra()
        with initialize_config_dir(version_base=None, config_dir=_config_dir()):
            cfg = compose(config_name="pretrain", overrides=["batch_size=16", "n_layer=2"])
            assert cfg.batch_size == 16
            assert cfg.n_layer == 2

    def test_sft_config_has_required_missing_keys(self):
        _clear_hydra()
        with initialize_config_dir(version_base=None, config_dir=_config_dir()):
            cfg = compose(config_name="sft")
            assert OmegaConf.is_missing(cfg, "init_from")
            assert cfg.out_dir == "out/sft"

    def test_generate_config_is_list_for_max_new_tokens(self):
        _clear_hydra()
        with initialize_config_dir(version_base=None, config_dir=_config_dir()):
            cfg = compose(config_name="generate")
            assert isinstance(cfg.max_new_tokens, (list, tuple, ListConfig))
            assert cfg.max_new_tokens[0] == 400

    def test_hydra_config_matches_legacy_pretrain_yaml(self, monkeypatch):
        """Hydra pretrain defaults should match the legacy pretrain parser + YAML."""
        _clear_hydra()

        from training.train_pretrain import get_args

        project_root = os.path.dirname(os.path.dirname(__file__))
        legacy_cfg = os.path.join(project_root, "config", "pretrain.yaml")
        monkeypatch.setattr(sys, "argv", ["train_pretrain.py", "--config", legacy_cfg])
        real_args = get_args()

        with initialize_config_dir(version_base=None, config_dir=_config_dir()):
            cfg = compose(config_name="pretrain")
            ns = to_namespace(cfg)

        assert ns.model == real_args.model
        assert ns.out_dir == real_args.out_dir
        assert ns.block_size == real_args.block_size
        assert ns.n_layer == real_args.n_layer
        assert ns.n_head == real_args.n_head
        assert ns.n_embd == real_args.n_embd
        assert ns.n_kv_head == real_args.n_kv_head
        assert ns.dropout == real_args.dropout
        assert ns.batch_size == real_args.batch_size
        assert ns.learning_rate == real_args.learning_rate
        assert ns.max_iters == real_args.max_iters
        assert ns.warmup_iters == real_args.warmup_iters
        assert ns.lr_decay_iters == real_args.lr_decay_iters
        assert ns.min_lr == real_args.min_lr
        assert ns.lr_schedule == real_args.lr_schedule
        assert ns.weight_decay == real_args.weight_decay
        assert ns.grad_clip == real_args.grad_clip
        assert ns.eval_interval == real_args.eval_interval
        assert ns.eval_iters == real_args.eval_iters
        assert ns.log_interval == real_args.log_interval
        assert ns.seed == real_args.seed
        assert ns.device == real_args.device
        assert ns.compile == real_args.compile
        assert ns.backend == real_args.backend
        assert ns.fsdp == real_args.fsdp
        assert ns.fsdp_sharding_strategy == real_args.fsdp_sharding_strategy
        assert ns.num_workers == real_args.num_workers
        assert ns.gradient_accumulation_steps == real_args.gradient_accumulation_steps
        assert ns.use_wandb == real_args.use_wandb
        assert ns.use_ema == real_args.use_ema
        assert ns.ema_decay == real_args.ema_decay
        assert ns.early_stopping_patience == real_args.early_stopping_patience
        assert ns.keep_last_n == real_args.keep_last_n
        assert ns.use_packing == real_args.use_packing
        assert ns.gradient_checkpointing == real_args.gradient_checkpointing
        assert ns.shuffle_buffer == real_args.shuffle_buffer
        assert ns.resume == real_args.resume
        assert ns.attn_backend == real_args.attn_backend
        assert ns.use_ring_attention == real_args.use_ring_attention
        assert ns.ring_block_size_q == real_args.ring_block_size_q
        assert ns.ring_block_size_kv == real_args.ring_block_size_kv
