"""
Unified configuration loading.

Training scripts can accept a YAML config file via ``--config``; values in the
file become argparse defaults, and command-line arguments still override them.
Nested YAML dictionaries are supported both in the file and on the command line
(``--optimizer.lr 1e-4``).
"""
import argparse
import os
from pathlib import Path

try:
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover - omegaconf is optional for the legacy path
    DictConfig = None  # type: ignore[misc,assignment]
    OmegaConf = None  # type: ignore[misc,assignment]


def _expand_env(value):
    """Recursively expand ``${VAR}`` placeholders in strings/lists/dicts."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def load_yaml_config(path):
    """Load a YAML config file, stripping a leading BOM if present."""
    import yaml
    text = Path(path).read_text(encoding="utf-8-sig")
    cfg = yaml.safe_load(text) or {}
    return _expand_env(cfg)


def _flatten(d, parent_key="", sep="."):
    """Flatten a nested dict into dot-separated keys."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def _unflatten(d, sep="."):
    """Turn dot-separated keys into a nested dict."""
    nested = {}
    for k, v in d.items():
        parts = k.split(sep)
        cur = nested
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = v
    return nested


def flatten(d, sep="."):
    """Public helper to flatten a nested dict."""
    return _flatten(d, sep=sep)


def unflatten(d, sep="."):
    """Public helper to unflatten a dict with dotted keys."""
    return _unflatten(d, sep=sep)


def _coerce_value(value):
    """Convert a CLI string to a primitive type when possible."""
    if not isinstance(value, str):
        return value
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


class NestedNamespace:
    """Argument namespace that supports nested attribute access.

    A config such as ``{'optimizer': {'lr': 1e-4}}`` can be accessed both as
    ``args.optimizer.lr`` and via ``args.get('optimizer.lr')``.
    """

    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, NestedNamespace(v))
            else:
                setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None, sep="."):
        """Get a possibly nested value by dotted key."""
        parts = key.split(sep)
        cur = self
        for part in parts:
            if isinstance(cur, NestedNamespace):
                cur = cur.__dict__.get(part, default)
            elif isinstance(cur, dict):
                cur = cur.get(part, default)
            else:
                return default
            if cur is default:
                return default
        return cur

    def to_dict(self):
        """Convert the namespace back to a nested dict."""
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, NestedNamespace):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result


def _namespace_to_nested(ns):
    """Convert a flat argparse Namespace into a NestedNamespace.

    Existing dict-valued attributes (e.g. from nested YAML defaults) are
    wrapped recursively.
    """
    d = vars(ns)
    converted = {}
    for k, v in d.items():
        if isinstance(v, dict):
            converted[k] = NestedNamespace(v)
        else:
            converted[k] = v
    return NestedNamespace(converted)


def _extract_nested_overrides(argv, sep="."):
    """Extract ``--a.b value`` overrides from argv and return (overrides, filtered_argv).

    Supports both ``--key value`` and ``--key=value`` forms.
    """
    overrides = {}
    filtered = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--"):
            key_part = arg[2:]
            if sep in key_part:
                if "=" in arg:
                    _, val = arg.split("=", 1)
                    overrides[key_part] = _coerce_value(val)
                    i += 1
                    continue
                if i + 1 < len(argv):
                    overrides[key_part] = _coerce_value(argv[i + 1])
                    i += 2
                    continue
                overrides[key_part] = True
                i += 1
                continue
        filtered.append(arg)
        i += 1
    return overrides, filtered


def _set_nested(ns, key, value, sep="."):
    """Set a possibly nested value on a NestedNamespace."""
    parts = key.split(sep)
    cur = ns
    for part in parts[:-1]:
        nxt = cur.__dict__.get(part)
        if not isinstance(nxt, NestedNamespace):
            nxt = NestedNamespace({})
            setattr(cur, part, nxt)
        cur = nxt
    setattr(cur, parts[-1], value)


def parse_args_with_config(parser: argparse.ArgumentParser, argv=None, sep="."):
    """
    Parse ``argv`` while respecting an optional ``--config`` YAML file.

    The config file values are applied as parser defaults, so explicit CLI
    arguments always win.  Nested YAML dictionaries are supported both in the
    file and on the command line via dot notation (``--optimizer.lr 1e-4``).

    Returns a :class:`NestedNamespace` with ``to_dict()`` and ``get(key)`` helpers.
    """
    # First, extract just the --config flag without interfering with the main parser.
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    pre_args, remaining = config_parser.parse_known_args(argv)

    config = {}
    if pre_args.config:
        if not Path(pre_args.config).exists():
            raise FileNotFoundError(f"Config file not found: {pre_args.config}")
        config = load_yaml_config(pre_args.config)

    # Pull out nested overrides so argparse doesn't reject unknown dotted keys.
    overrides, filtered = _extract_nested_overrides(remaining, sep=sep)

    # Apply config defaults; existing argparse defaults remain for keys not present.
    parser.set_defaults(**config)
    args = parser.parse_args(filtered)

    # Wrap dict-valued defaults and apply nested CLI overrides.
    ns = _namespace_to_nested(args)
    for key, value in overrides.items():
        _set_nested(ns, key, value, sep=sep)

    return ns


def to_dict(args, sep="."):
    """Convert an argparse Namespace, NestedNamespace or DictConfig to a nested dict."""
    if DictConfig is not None and isinstance(args, DictConfig):
        return OmegaConf.to_container(args, resolve=True)
    if hasattr(args, "to_dict"):
        return args.to_dict()
    d = vars(args) if hasattr(args, "__dict__") else dict(args)
    # If the namespace was flattened to dotted keys, unflatten it.
    if any(sep in k for k in d.keys()):
        return _unflatten(d, sep=sep)
    return d


def validate_keys(args, required, sep="."):
    """Raise ValueError if any required dotted key is missing or None."""
    missing = []
    for key in required:
        value = args.get(key, sep=sep) if hasattr(args, "get") else _unflatten(vars(args), sep=sep).get(key)
        if value is None:
            missing.append(key)
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    return True
