"""Helpers for integrating Hydra/OmegaConf with the legacy argparse pipeline.

The project still relies on ``BaseTrainer`` (and many training loops) receiving
an attribute-accessible object that supports ``vars(args)`` and
``utils.config.to_dict(args)``.  These helpers convert an OmegaConf
``DictConfig`` into the existing :class:`utils.config.NestedNamespace` so that
new Hydra entry points can reuse the old trainers without modification.
"""
from __future__ import annotations

from typing import Any, List, Optional, Set

from omegaconf import DictConfig, ListConfig, OmegaConf

from utils.config import NestedNamespace


def _to_container(value: Any) -> Any:
    """Recursively turn OmegaConf containers into plain Python objects."""
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, ListConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def to_namespace(cfg: DictConfig) -> NestedNamespace:
    """Convert a Hydra/OmegaConf ``DictConfig`` to a ``NestedNamespace``.

    The returned object supports attribute access (``args.batch_size``),
    dotted access (``args.optimizer.lr`` via :meth:`NestedNamespace.get`),
    ``vars(args)`` and ``utils.config.to_dict(args)``.  This lets the legacy
    ``BaseTrainer`` code consume Hydra configs unchanged.

    Parameters
    ----------
    cfg
        A ``DictConfig`` (typically the one passed to ``@hydra.main``).

    Returns
    -------
    NestedNamespace
        Attribute-accessible wrapper around the resolved configuration.
    """
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"to_namespace expects a DictConfig, got {type(cfg).__name__}")
    plain = _to_container(cfg)
    if not isinstance(plain, dict):
        raise TypeError(f"Root config must be a dict, got {type(plain).__name__}")
    return NestedNamespace(plain)


def missing_keys(cfg: DictConfig) -> List[str]:
    """Return the list of missing mandatory keys in ``cfg``.

    Uses ``OmegaConf.missing_keys`` recursively so that nested missing values
    are reported with dotted paths (e.g. ``optimizer.lr``).
    """
    return list(OmegaConf.missing_keys(cfg))


def validate_required(cfg: DictConfig, required: Optional[Set[str]] = None) -> None:
    """Raise ``ValueError`` if any required key is missing.

    Parameters
    ----------
    cfg
        Hydra configuration to validate.
    required
        Set of dotted keys that must be present.  If ``None``, only keys that
        are explicitly marked ``???`` in the config are checked.
    """
    if required:
        missing = [k for k in required if OmegaConf.is_missing(cfg, k)]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")
    else:
        miss = missing_keys(cfg)
        if miss:
            raise ValueError(f"Missing required config keys: {miss}")
