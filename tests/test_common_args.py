"""FR11a (T017): training/args.py common trainer flag extraction.

Locks the contract for ``training.args.add_common_trainer_args(parser)``:

- a single, side-effect-free assembly function owns the 20 flags that
  are shared across the five trainer scripts (train_pretrain /
  train_sft / train_grpo / train_dpo / iterative_grpo);
- every script must expose those flags with identical defaults, types
  and --help text;
- ``--ckpt_interval`` (FR3) must be injected into all five scripts
  (currently only train_pretrain defines it);
- script-specific defaults (batch_size, learning_rate, out_dir, ...)
  must remain untouched;
- the five scripts must stop re-declaring the common flags, i.e. the
  aggregate ``add_argument`` count in training/ drops sharply.
"""

import importlib
import inspect
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

TRAINER_MODULES = [
    "training.train_pretrain",
    "training.train_sft",
    "training.train_grpo",
    "training.train_dpo",
    "training.iterative_grpo",
]

# Flags that every trainer must expose through add_common_trainer_args.
# (flag name, default) -- types are asserted separately.
COMMON_FLAGS = {
    "seed": 1337,
    "backend": "nccl",
    "grad_clip": 1.0,
    "use_wandb": False,
    "exp_id": None,
    "no_metrics": False,
    "no_monitoring": False,
    "resume": None,
    "config": None,
    "ckpt_interval": 0,
    "gradient_accumulation_steps": 1,
    "lr_schedule": "cosine",
    "keep_last_n": 0,
    "attn_backend": "auto",
    "group_size": 4,
    "num_steps": 1000,
    "max_prompt_len": 128,
    "max_response_len": 128,
    "eps": 0.2,
}

# Typed flags: (flag, cli value, expected python type)
TYPED_FLAGS = [
    ("seed", "42", int),
    ("grad_clip", "0.5", float),
    ("ckpt_interval", "17", int),
    ("backend", "gloo", str),
    ("lr_schedule", "wsd", str),
    ("keep_last_n", "3", int),
    ("eps", "0.3", float),
    ("group_size", "8", int),
]

# Scripts whose get_args() needs mandatory positional-ish flags.
REQUIRED_ARGV = {
    "training.train_sft": ["--init_from", "dummy.pt"],
    "training.train_grpo": ["--init_from", "dummy.pt", "--ref_from", "ref.pt"],
    "training.train_dpo": ["--init_from", "dummy.pt"],
    "training.iterative_grpo": ["--init_from", "dummy.pt", "--ref_from", "ref.pt"],
}

LR_SCHEDULE_CHOICES = ["cosine", "linear", "wsd", "constant"]
ATTN_BACKEND_CHOICES = ["auto", "flash", "mem_efficient", "math", "default"]


def _get_args(modname, monkeypatch, argv):
    mod = importlib.import_module(modname)
    monkeypatch.setattr(sys, "argv", ["prog"] + argv)
    return mod.get_args()


# ---------------------------------------------------------------------------
# 1. The shared module itself
# ---------------------------------------------------------------------------


def test_training_args_module_exposes_add_common_trainer_args():
    args_mod = importlib.import_module("training.args")
    assert callable(getattr(args_mod, "add_common_trainer_args"))


def test_add_common_trainer_args_is_pure_assembly():
    """The helper must assemble the parser; it must never parse."""
    args_mod = importlib.import_module("training.args")
    src = inspect.getsource(args_mod.add_common_trainer_args)
    assert "parse_args" not in src


def test_add_common_trainer_args_registers_all_common_flags():
    import argparse

    args_mod = importlib.import_module("training.args")
    parser = argparse.ArgumentParser()
    args_mod.add_common_trainer_args(parser)
    ns = parser.parse_args([])
    for flag, default in COMMON_FLAGS.items():
        assert hasattr(ns, flag), flag
        assert getattr(ns, flag) == default, flag


def test_add_common_trainer_args_types_and_choices():
    import argparse

    args_mod = importlib.import_module("training.args")
    parser = argparse.ArgumentParser()
    args_mod.add_common_trainer_args(parser)

    argv = []
    for flag, value, _typ in TYPED_FLAGS:
        argv.extend(["--%s" % flag, value])
    ns = parser.parse_args(argv)
    for flag, value, typ in TYPED_FLAGS:
        got = getattr(ns, flag)
        assert isinstance(got, typ), (flag, type(got))
        if typ is float:
            assert got == pytest.approx(float(value)), flag
        else:
            assert got == typ(value), flag

    # store_true flags must stay boolean-toggled
    ns2 = parser.parse_args(["--use_wandb", "--no_metrics", "--no_monitoring"])
    assert ns2.use_wandb is True
    assert ns2.no_metrics is True
    assert ns2.no_monitoring is True

    # choices contracts
    with pytest.raises(SystemExit):
        parser.parse_args(["--lr_schedule", "bogus"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--attn_backend", "bogus"])


# ---------------------------------------------------------------------------
# 2. All five trainer scripts expose the common flags identically
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modname", TRAINER_MODULES)
def test_five_trainers_expose_common_flag_defaults(modname, monkeypatch):
    argv = list(REQUIRED_ARGV.get(modname, []))
    ns = _get_args(modname, monkeypatch, argv)
    for flag, default in COMMON_FLAGS.items():
        assert hasattr(ns, flag), (modname, flag)
        assert getattr(ns, flag) == default, (modname, flag)


@pytest.mark.parametrize("modname", TRAINER_MODULES)
def test_ckpt_interval_injected_everywhere(modname, monkeypatch):
    """FR11 acceptance: --ckpt_interval shows up in all five scripts."""
    argv = list(REQUIRED_ARGV.get(modname, []))
    ns = _get_args(modname, monkeypatch, argv)
    assert getattr(ns, "ckpt_interval", None) == 0
    ns2 = _get_args(modname, monkeypatch, argv + ["--ckpt_interval", "25"])
    assert ns2.ckpt_interval == 25


@pytest.mark.parametrize("modname", TRAINER_MODULES)
def test_store_true_semantics_preserved(modname, monkeypatch):
    argv = list(REQUIRED_ARGV.get(modname, []))
    ns = _get_args(
        modname,
        monkeypatch,
        argv + ["--use_wandb", "--no_metrics", "--no_monitoring"],
    )
    assert ns.use_wandb is True
    assert ns.no_metrics is True
    assert ns.no_monitoring is True


@pytest.mark.parametrize("modname", TRAINER_MODULES)
def test_common_help_text_lists_all_common_flags(modname, monkeypatch, capsys):
    """--help must list every common flag for every script."""
    blocks = _collect_help_blocks(modname, monkeypatch, capsys)
    for flag in COMMON_FLAGS:
        assert flag in blocks, (modname, flag)


def test_common_help_blocks_identical_across_trainers(monkeypatch, capsys):
    """The --help block of every common flag must match across scripts."""
    all_blocks = {
        m: _collect_help_blocks(m, monkeypatch, capsys) for m in TRAINER_MODULES
    }
    base = all_blocks[TRAINER_MODULES[0]]
    for flag in COMMON_FLAGS:
        for modname in TRAINER_MODULES[1:]:
            assert all_blocks[modname][flag] == base[flag], (modname, flag)


def _collect_help_blocks(modname, monkeypatch, capsys):
    argv = list(REQUIRED_ARGV.get(modname, [])) + ["--help"]
    monkeypatch.setattr(sys, "argv", ["prog"] + argv)
    mod = importlib.import_module(modname)
    with pytest.raises(SystemExit) as excinfo:
        mod.get_args()
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    return _extract_help_blocks(out)


def _extract_help_blocks(help_text):
    """Map '--flag' -> its full help block from an argparse --help dump."""
    lines = help_text.splitlines()
    starts = [i for i, ln in enumerate(lines) if re.match(r"^\s+-{1,2}\w", ln)]
    blocks = {}
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        chunk = "\n".join(lines[start:end]).rstrip()
        m = re.match(r"^\s+--([\w-]+)", chunk)
        if m:
            blocks[m.group(1)] = chunk
    return blocks


# ---------------------------------------------------------------------------
# 3. Script-specific defaults must survive the extraction
# ---------------------------------------------------------------------------


def test_script_specific_defaults_preserved(monkeypatch):
    pt = _get_args("training.train_pretrain", monkeypatch, [])
    assert pt.batch_size == 12
    assert pt.out_dir == "out/pretrain"
    assert pt.learning_rate == pytest.approx(6e-4)
    assert pt.min_lr == pytest.approx(6e-5)
    assert pt.eval_interval == 1000
    assert pt.model == "modern"

    sft = _get_args(
        "training.train_sft", monkeypatch, ["--init_from", "dummy.pt"]
    )
    assert sft.batch_size == 8
    assert sft.out_dir == "out/sft"
    assert sft.learning_rate == pytest.approx(3e-4)
    assert sft.variant == "sft-only"

    gr = _get_args(
        "training.train_grpo",
        monkeypatch,
        ["--init_from", "d.pt", "--ref_from", "r.pt"],
    )
    assert gr.batch_size == 4
    assert gr.weight_decay == pytest.approx(0.01)
    assert gr.beta == pytest.approx(0.04)
    assert gr.adv_norm == "group"

    dp = _get_args("training.train_dpo", monkeypatch, ["--init_from", "d.pt"])
    assert dp.learning_rate == pytest.approx(1e-6)
    assert dp.min_lr == pytest.approx(1e-7)
    assert dp.eval_interval == 50
    assert dp.beta == pytest.approx(0.1)
    assert dp.preference_loss == "dpo"

    ig = _get_args(
        "training.iterative_grpo",
        monkeypatch,
        ["--init_from", "d.pt", "--ref_from", "r.pt"],
    )
    assert ig.out_dir == "out/iterative_grpo"
    assert ig.rejection_interval == 200
    assert ig.ref_update_interval == 250


@pytest.mark.parametrize(
    "modname,flag",
    [
        ("training.train_pretrain", "fsdp"),
        ("training.train_pretrain", "use_ema"),
        ("training.train_pretrain", "use_packing"),
        ("training.train_sft", "variant"),
        ("training.train_grpo", "adv_norm"),
        ("training.train_grpo", "prompt_diversity"),
        ("training.train_dpo", "preference_loss"),
        ("training.train_dpo", "win_rate_eval"),
        ("training.iterative_grpo", "rejection_interval"),
        ("training.iterative_grpo", "ref_update_ratio"),
    ],
)
def test_unique_flags_still_present(modname, flag, monkeypatch):
    argv = list(REQUIRED_ARGV.get(modname, []))
    ns = _get_args(modname, monkeypatch, argv)
    assert hasattr(ns, flag), (modname, flag)


# ---------------------------------------------------------------------------
# 4. De-duplication: the common flags move out of the five scripts
# ---------------------------------------------------------------------------


def _count_add_argument(path):
    src = path.read_text(encoding="utf-8")
    return len(re.findall(r"\.add_argument\(", src))


def test_common_flags_not_redeclared_in_scripts():
    """Green state: scripts delegate common flags to training/args.py."""
    ban = [
        "--seed",
        "--backend",
        "--grad_clip",
        "--use_wandb",
        "--no_metrics",
        "--no_monitoring",
        "--resume",
        "--config",
        "--ckpt_interval",
    ]
    for name in [
        "train_pretrain.py",
        "train_sft.py",
        "train_grpo.py",
        "train_dpo.py",
        "iterative_grpo.py",
    ]:
        src = (REPO_ROOT / "training" / name).read_text(encoding="utf-8")
        for flag in ban:
            assert ('"%s"' % flag) not in src, (name, flag)


def test_total_add_argument_count_dropped():
    """Five scripts declare <= 115 flags total (baseline: 185)."""
    total = sum(
        _count_add_argument(REPO_ROOT / "training" / n)
        for n in [
            "train_pretrain.py",
            "train_sft.py",
            "train_grpo.py",
            "train_dpo.py",
            "iterative_grpo.py",
        ]
    )
    assert total <= 115, total


def test_all_scripts_call_add_common_trainer_args():
    for name in [
        "train_pretrain.py",
        "train_sft.py",
        "train_grpo.py",
        "train_dpo.py",
        "iterative_grpo.py",
    ]:
        src = (REPO_ROOT / "training" / name).read_text(encoding="utf-8")
        assert "add_common_trainer_args(" in src, name
