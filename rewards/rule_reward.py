"""
Rule-based reward function for arithmetic alignment.

Provides continuous, fine-grained rewards:

* format score      : proper use of ``<answer>...</answer>`` tags
* process score     : presence of intermediate derivation / reasoning steps
* accuracy score    : continuous partial credit based on relative error

The total reward is the sum of the three components and lies in ``[0.0, 2.0]``.
"""
import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


# Regex for <answer>...</answer> blocks (non-greedy, dotall).
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# Tokens that suggest the model showed its work / intermediate steps.
_DERIVATION_MARKERS = re.compile(
    r"(=\s*[\d\-\.]|\bstep\s+\d|\bfirst\b|\bthen\b|\bnext\b|\bso\b|"
    r"\btherefore\b|=>|->|\+\s*|\-\s*|\*\s*|/\s*|\*\*\s*|%\s*)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedAnswer:
    """Parsed result from a model response."""

    has_open_tag: bool = False
    has_close_tag: bool = False
    num_blocks: int = 0
    value: Optional[float] = None
    raw_content: str = ""


def _normalize_number_string(s: str) -> str:
    """Strip common formatting artifacts (commas, spaces, currency symbols)."""
    s = s.strip()
    # Remove common thousands separators and whitespace.
    s = s.replace(",", "").replace(" ", "").replace("$", "").replace("\u00a0", "")
    # Strip a trailing period or stray punctuation that may follow a number.
    s = s.rstrip(".!?;:")
    return s


def _parse_number(s: str) -> Optional[float]:
    """Parse a numeric string, rejecting nan/inf/overflow."""
    try:
        s = _normalize_number_string(s)
        if not s:
            return None
        val = float(s)
        if not math.isfinite(val):
            return None
        return val
    except (ValueError, TypeError, OverflowError):
        return None


def parse_answer(text: str) -> ParsedAnswer:
    """Extract and validate ``<answer>...</answer>`` blocks from a response.

    Returns
    -------
    ParsedAnswer
        ``num_blocks`` counts complete blocks; ``value`` is the parsed numeric
        content of the *first* complete block, or ``None`` if it is not a valid
        finite number.
    """
    has_open = "<answer>" in text
    has_close = "</answer>" in text

    matches = list(_ANSWER_RE.finditer(text))
    if not matches:
        return ParsedAnswer(has_open_tag=has_open, has_close_tag=has_close)

    first = matches[0]
    content = first.group(1).strip()
    value = _parse_number(content)
    return ParsedAnswer(
        has_open_tag=True,
        has_close_tag=True,
        num_blocks=len(matches),
        value=value,
        raw_content=content,
    )


def _format_score(parsed: ParsedAnswer) -> float:
    """Reward proper answer formatting.

    Scoring:
    * 0.5 : exactly one complete ``<answer>number</answer>`` block
    * 0.3 : one or more blocks but malformed in some way (non-numeric content,
            mismatched tags, or multiple blocks)
    * 0.0 : no answer block at all
    """
    if parsed.num_blocks == 1 and parsed.value is not None:
        return 0.5
    if parsed.num_blocks >= 1 or parsed.has_open_tag or parsed.has_close_tag:
        # Some effort to use the tag, but malformed.
        return 0.3
    return 0.0


def _process_score(response: str, parsed: ParsedAnswer) -> float:
    """Reward intermediate derivation / reasoning steps.

    The score is based on the length and presence of mathematical markers
    *outside* the final answer block (so that models are encouraged to show
    their work, not just emit the answer).
    """
    if parsed.num_blocks == 1:
        # Consider only the text before the first answer block.
        prefix = response.split("<answer>", 1)[0]
    else:
        prefix = response

    prefix = prefix.strip()
    if not prefix:
        return 0.0

    # Count derivation markers.
    marker_count = len(_DERIVATION_MARKERS.findall(prefix))
    # Reward up to 0.3 based on marker density and presence.
    if marker_count == 0:
        return 0.0
    if marker_count >= 4:
        return 0.3
    if marker_count >= 2:
        return 0.2
    return 0.1


def _relative_error(pred: float, ref: float) -> float:
    """Return relative error; handle ref == 0 with absolute error."""
    if ref == 0.0:
        return abs(pred)
    return abs(pred - ref) / abs(ref)


def _accuracy_score(pred: Optional[float], ref: Optional[float]) -> float:
    """Continuous accuracy reward based on relative error.

    Scoring (relative error):
    * < 1e-6 : 1.2  (effectively exact)
    * < 1e-4 : 0.9  (very close)
    * < 1e-2 : 0.6  (close)
    * < 1e-1 : 0.3  (approximate)
    * otherwise : 0.0

    This gives a non-zero gradient even when the answer is not perfectly exact.
    """
    if pred is None or ref is None:
        return 0.0
    if not (math.isfinite(pred) and math.isfinite(ref)):
        return 0.0

    rel_err = _relative_error(pred, ref)
    if rel_err < 1e-6:
        return 1.2
    if rel_err < 1e-4:
        return 0.9
    if rel_err < 1e-2:
        return 0.6
    if rel_err < 1e-1:
        return 0.3
    return 0.0


def compute_reward(response: str, reference: str) -> Tuple[float, float, float, float]:
    """Compute fine-grained reward for a single response.

    Parameters
    ----------
    response : str
        Model-generated text.
    reference : str
        Ground-truth answer (numeric string).

    Returns
    -------
    total_reward : float
        Sum of format, process, and accuracy components (range ``[0.0, 2.0]``).
    format_score : float
        Reward for using ``<answer>...</answer>`` correctly.
    process_score : float
        Reward for showing intermediate work.
    accuracy_score : float
        Reward based on numeric closeness to the reference.
    """
    parsed = parse_answer(response)
    ref_val = _parse_number(reference)

    fmt = _format_score(parsed)
    proc = _process_score(response, parsed)
    acc = _accuracy_score(parsed.value, ref_val)

    total = fmt + proc + acc
    return total, fmt, proc, acc


def compute_reward_batch(
    responses: List[str], references: List[str]
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Batch version of :func:`compute_reward`.

    Returns
    -------
    rewards, format_scores, process_scores, accuracy_scores : list[float]
    """
    rewards = []
    fmt_scores = []
    proc_scores = []
    acc_scores = []
    for resp, ref in zip(responses, references):
        r, f, p, a = compute_reward(resp, ref)
        rewards.append(r)
        fmt_scores.append(f)
        proc_scores.append(p)
        acc_scores.append(a)
    return rewards, fmt_scores, proc_scores, acc_scores
