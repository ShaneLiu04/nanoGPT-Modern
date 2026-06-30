"""Multi-source dataset mixing utilities.

Provides both a standalone mixing strategy and a ``torch.utils.data.IterableDataset``
that samples from multiple source datasets according to configured proportions.
"""

from __future__ import annotations

import json
import warnings
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import numpy as np


class MixtureStrategy:
    """Convert raw mixture weights into sampling probabilities.

    A temperature ``t`` is applied as ``p_i = w_i^(1/t) / sum(...)`` so that
    ``t > 1`` makes the distribution more uniform and ``t < 1`` makes it more
    peaked.
    """

    def __init__(
        self,
        weights: Mapping[str, float],
        temperature: float = 1.0,
    ):
        if not weights:
            raise ValueError("weights must not be empty")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.names = list(weights.keys())
        raw = np.array([float(weights[name]) for name in self.names], dtype=np.float64)
        if raw.sum() <= 0:
            raise ValueError("weights must sum to a positive value")
        # Apply temperature scaling.
        if temperature != 1.0:
            raw = np.power(raw, 1.0 / temperature)
        self.probabilities = raw / raw.sum()
        self.temperature = temperature

    def sample_source(self, rng: np.random.Generator) -> str:
        """Sample a source name according to the mixture probabilities."""
        return rng.choice(self.names, p=self.probabilities)

    def as_dict(self) -> Dict[str, float]:
        """Return the final sampling probabilities as a dict."""
        return dict(zip(self.names, self.probabilities.tolist()))


class MixedIterableDataset:
    """Iterate over multiple source datasets with a fixed mixture.

    Each source can be any iterable that yields examples.  The mixer samples a
    source on every step, then yields the next example from an internal
    iterator for that source.  Sources that are exhausted are removed from the
    mixture unless ``stop_on_exhaustion=True``.

    Parameters
    ----------
    sources:
        Mapping from source name to iterable of examples.
    weights:
        Raw sampling weights per source.
    temperature:
        Mixture temperature.
    total_examples:
        Total number of examples to yield.  ``None`` means run until all
        sources are exhausted.
    seed:
        Random seed for reproducibility.
    stop_on_exhaustion:
        If ``True``, stop as soon as any source runs out.  Useful for aligned
        epoch boundaries.
    """

    def __init__(
        self,
        sources: Mapping[str, Iterable[Any]],
        weights: Mapping[str, float],
        temperature: float = 1.0,
        total_examples: Optional[int] = None,
        seed: int = 0,
        stop_on_exhaustion: bool = False,
    ):
        self.strategy = MixtureStrategy(weights, temperature)
        if set(sources.keys()) != set(self.strategy.names):
            raise ValueError("keys of sources and weights must match")
        self._sources = dict(sources)
        self._iterators: Dict[str, Iterator[Any]] = {}
        self._total = total_examples
        self._rng = np.random.default_rng(seed)
        self._stop_on_exhaustion = stop_on_exhaustion
        self._yielded = 0

    def __iter__(self) -> Iterator[Any]:
        self._iterators = {name: iter(src) for name, src in self._sources.items()}
        active = set(self._iterators.keys())
        resume = self._yielded
        yielded = 0
        while (self._total is None or yielded < self._total) and active:
            name = self.strategy.sample_source(self._rng)
            if name not in active:
                continue
            try:
                example = next(self._iterators[name])
            except StopIteration:
                if self._stop_on_exhaustion:
                    return
                active.remove(name)
                continue
            if isinstance(example, dict):
                example["__mix_source__"] = name
            yielded += 1
            if yielded <= resume:
                # Skip already-consumed examples while keeping the RNG state
                # in sync with the original run.
                continue
            self._yielded = yielded
            yield example

    def state_dict(self) -> Dict[str, Any]:
        """Return a serializable state for resuming.

        The RNG state is captured so that resuming reproduces the same mixture
        sampling sequence after skipping the already-yielded examples.
        """
        return {
            "yielded": self._yielded,
            "total": self._total,
            "probabilities": self.strategy.as_dict(),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the iterator state from a checkpoint."""
        self._yielded = int(state.get("yielded", 0))
        self._total = state.get("total", self._total)
        probabilities = state.get("probabilities")
        if probabilities is not None:
            self.strategy = MixtureStrategy(probabilities, self.strategy.temperature)
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._rng.bit_generator.state = rng_state


def mix_datasets(
    datasets: Mapping[str, Any],
    weights: Mapping[str, float],
    temperature: float = 1.0,
    seed: int = 0,
) -> Any:
    """Interleave HuggingFace ``datasets.Dataset`` objects if possible.

    If ``datasets`` is importable and all sources are ``datasets.Dataset``
    instances, this delegates to ``datasets.interleave_datasets`` for a
    fully materialized mixed dataset.  Otherwise it returns a
    ``MixedIterableDataset``.
    """
    try:
        from datasets import Dataset, interleave_datasets  # type: ignore[import-untyped]

        if all(isinstance(ds, Dataset) for ds in datasets.values()):
            names = list(datasets.keys())
            ds_list = [datasets[name] for name in names]
            strategy = MixtureStrategy(
                {name: weights.get(name, 1.0) for name in names}, temperature
            )
            return interleave_datasets(
                ds_list,
                probabilities=strategy.probabilities.tolist(),
                seed=seed,
            )
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"Could not use datasets.interleave_datasets ({exc}); falling back to MixedIterableDataset."
        )

    return MixedIterableDataset(
        datasets,
        weights,
        temperature=temperature,
        seed=seed,
    )


def load_mixture_config(path: str) -> Dict[str, Any]:
    """Load a mixture config JSON such as::

    {
      "weights": {"openwebtext": 0.7, "wikipedia": 0.3},
      "temperature": 0.8,
      "seed": 42
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "weights" not in cfg:
        raise ValueError("mixture config must contain 'weights'")
    return cfg
