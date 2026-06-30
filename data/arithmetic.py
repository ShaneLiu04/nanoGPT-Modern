"""
Synthetic arithmetic dataset generator.
Three difficulty levels: easy, medium, hard.
"""

import ast
import operator
import random
import re
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

# Safe AST-based operators for arithmetic evaluation (replaces eval()).
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(expr):
    """Evaluate a simple arithmetic expression safely using AST.

    Supports +, -, *, /, **, %, parentheses and unary minus.
    Returns (value, None) on success or (None, error_message) on failure.
    Integer results are kept as integers; floating-point results are rounded
    to 6 decimal places to avoid scientific-notation noise in answer strings.
    """
    try:
        node = ast.parse(expr.strip(), mode="eval").body
        val = _eval_node(node)
        if isinstance(val, float) and not val.is_integer():
            return round(val, 6), None
        if isinstance(val, float) and val.is_integer():
            return int(val), None
        return val, None
    except ZeroDivisionError:
        return None, "division_by_zero"
    except OverflowError:
        return None, "overflow"
    except Exception as e:
        return None, str(e)


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Num):  # Python < 3.8 compatibility
        return node.n
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError(f"unsupported binary operator: {op_type.__name__}")
        return _ALLOWED_OPS[op_type](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError(f"unsupported unary operator: {op_type.__name__}")
        return _ALLOWED_OPS[op_type](_eval_node(node.operand))
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Prompt templating
# ---------------------------------------------------------------------------
# The original dataset used a single fixed instruction.  Realistic alignment
# training benefits from prompt diversity: it reduces overfitting to a single
# phrasing and improves generalization to natural-language task descriptions.
# Each template renders ``{expr}`` (the arithmetic expression) and always
# instructs the model to wrap the final answer in ``<answer>...</answer>``
# so the reward function (``rewards/rule_reward.py``) can parse it.
#
# Templates are split into "instructional" (explicit step-by-step) and
# "natural" (conversational) styles.  ``prompt_diversity`` controls how
# many templates are active:
#   0  -> original single template (backward compatible)
#   1  -> all templates, uniformly sampled
#   >1 -> N templates, deterministically chosen (deterministic subset)
PROMPT_TEMPLATES = [
    "Solve: {expr}\nWrap your final answer in <answer>...</answer>.",
    "Question: Compute {expr}.\nPlease think step by step and write your final answer inside <answer>...</answer>.",
    "What is the result of {expr}?\nShow your reasoning, then put the final number in <answer>...</answer>.",
    "Evaluate the expression: {expr}\nEnd your response with <answer>...</answer> containing the result.",
    "Problem: {expr}\nWork through the calculation carefully. Enclose the final answer in <answer>...</answer>.",
    "Calculate: {expr}\nReason briefly and wrap the answer in <answer>...</answer>.",
    "Please solve the following arithmetic problem:\n{expr}\nFinal answer must be written as <answer>...</answer>.",
    "Math: {expr} = ?\nThink step by step. The answer goes in <answer>...</answer>.",
    "Find the value of {expr}.\nShow your work. Put only the final value in <answer>...</answer>.",
    "Solve the problem below and box your answer with <answer>...</answer>.\n{expr}",
    "Compute {expr} step by step.\nAnswer format: <answer>...</answer>",
    "Task: Arithmetic\nInput: {expr}\nOutput your final answer in <answer>...</answer>.",
]
# Index 0 is the original template; keep it first for determinism.


def select_prompt_templates(diversity: int = 1):
    """Return the list of prompt templates to use given a diversity setting.

    Parameters
    ----------
    diversity : int
        ``0`` keeps only the original template (backward compatible).
        ``1`` (default) returns all available templates.
        ``N>1`` returns the first ``N`` templates in a deterministic order.
    """
    if diversity <= 0:
        return [PROMPT_TEMPLATES[0]]
    if diversity == 1:
        return list(PROMPT_TEMPLATES)
    return PROMPT_TEMPLATES[:diversity]


def format_prompt(
    expr: str, diversity: int = 1, rng: Optional[random.Random] = None
) -> str:
    """Format an arithmetic expression into a prompt using diverse templates.

    Parameters
    ----------
    expr      : The arithmetic expression string (e.g. ``"3 + 5 * 2"``).
    diversity : Prompt diversity level (see ``select_prompt_templates``).
    rng       : Optional ``random.Random`` instance for reproducibility.  When
                ``None``, falls back to the module-level ``random`` state.
    """
    templates = select_prompt_templates(diversity)
    if len(templates) == 1:
        return templates[0].format(expr=expr)
    chosen = rng.choice(templates) if rng is not None else random.choice(templates)
    return chosen.format(expr=expr)


def generate_easy(num_samples=1000, seed=42, prompt_diversity: int = 1):
    random.seed(seed)
    rng = random.Random(seed + 7)
    data: list[dict[str, Any]] = []
    ops = ["+", "-", "*", "/"]
    for _ in range(num_samples):
        a = random.randint(0, 1000)
        b = random.randint(1, 100)  # avoid div by 0
        op = random.choice(ops)
        ans: int | float
        if op == "+":
            ans = a + b
        elif op == "-":
            ans = a - b
        elif op == "*":
            ans = a * b
        else:
            ans = a / b
            ans = round(ans, 4)
        expr = f"{a} {op} {b}"
        prompt = format_prompt(expr, prompt_diversity, rng)
        data.append({"prompt": prompt, "answer": str(ans), "level": "easy"})
    return data


def generate_medium(num_samples=1000, seed=43, prompt_diversity: int = 1):
    """Medium arithmetic: 2-3 step mixed operations with optional parentheses."""
    random.seed(seed)
    rng = random.Random(seed + 7)
    data: list[dict[str, Any]] = []
    ops = ["+", "-", "*", "/"]
    attempts = 0
    while len(data) < num_samples and attempts < num_samples * 3:
        # Half the time: flat expression; half: one-group parentheses
        if random.random() < 0.5:
            steps = random.randint(2, 3)
            nums = [random.randint(0, 1000) for _ in range(steps + 1)]
            chosen_ops = [random.choice(ops) for _ in range(steps)]
            expr = str(nums[0])
            for i in range(steps):
                expr += f" {chosen_ops[i]} {nums[i + 1]}"
        else:
            a = random.randint(0, 500)
            b = random.randint(1, 500)
            c = random.randint(0, 500)
            d = random.randint(1, 500)
            op1 = random.choice(ops)
            op2 = random.choice(ops)
            # ensure no div-by-zero in either operand if it's a division
            if op1 == "/":
                b = random.randint(1, 100)
            if op2 == "/":
                c = random.randint(1, 100)
            expr = f"({a} {op1} {b}) {op2} {c}"

        ans, err = _safe_eval(expr)
        attempts += 1
        if err is not None:
            continue
        prompt = format_prompt(expr, prompt_diversity, rng)
        data.append({"prompt": prompt, "answer": str(ans), "level": "medium"})
    return data


def generate_hard(num_samples=1000, seed=44, prompt_diversity: int = 1):
    """Diverse hard arithmetic problems with multiple templates.

    Templates (equally weighted):
      - ``(A + B) * C / D``       nested parens
      - ``A + B * C - D / E``     operator precedence
      - ``(A * B) + (C / D)``     multi-group
      - ``A ** B % C``            exponent + modulo
      - ``(A - B) * (C + D)``    two-group multiplication
    """
    random.seed(seed)
    rng = random.Random(seed + 7)
    data: list[dict[str, Any]] = []

    templates = [
        lambda: (
            f"({random.randint(0, 10000)} + {random.randint(1, 10000)}) * "
            f"{random.randint(1, 500)} / {random.randint(1, 500)}"
        ),
        lambda: (
            f"{random.randint(0, 5000)} + {random.randint(1, 100)} * "
            f"{random.randint(1, 100)} - {random.randint(1, 500)} / {random.randint(1, 100)}"
        ),
        lambda: (
            f"({random.randint(0, 5000)} * {random.randint(1, 100)}) + "
            f"({random.randint(1, 1000)} / {random.randint(1, 100)})"
        ),
        lambda: f"{random.randint(2, 20)} ** {random.randint(1, 6)} % {random.randint(1, 50) + 1}",
        lambda: (
            f"({random.randint(0, 5000)} - {random.randint(0, 5000)}) * "
            f"({random.randint(1, 200)} + {random.randint(1, 200)})"
        ),
    ]

    attempts = 0
    while len(data) < num_samples and attempts < num_samples * 3:
        template = random.choice(templates)
        expr = template()
        ans, err = _safe_eval(expr)
        attempts += 1
        if err is not None:
            continue
        prompt = format_prompt(expr, prompt_diversity, rng)
        data.append({"prompt": prompt, "answer": str(ans), "level": "hard"})
    return data


class ArithmeticDataset(Dataset):
    """Arithmetic dataset with optional up-front tokenization.

    Parameters
    ----------
    data : list[dict]
        Output of ``generate_easy/medium/hard``.
    tokenizer : tiktoken.Encoding
    max_length : int
    pre_tokenize : bool
        If True (default), encode all samples in ``__init__`` to avoid
        redundant tokenizer calls inside DataLoader workers.
    """

    def __init__(self, data, tokenizer, max_length=256, pre_tokenize=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pre_tokenize = pre_tokenize

        if pre_tokenize:
            self.samples = [self._encode_item(item) for item in data]
        else:
            self.samples = None

    def _encode_item(self, item):
        prompt = item["prompt"]
        answer = item["answer"]
        # Consistent format: no extra space before <answer>.
        full_text = prompt + "<answer>" + answer + "</answer>"
        tokens = self.tokenizer.encode(full_text)
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt": prompt,
            "answer": answer,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.samples is not None:
            return self.samples[idx]
        return self._encode_item(self.data[idx])


# ---------------------------------------------------------------------------
# Chain-of-Thought (CoT) extensions
# ---------------------------------------------------------------------------


def generate_cot_problems(num_samples=1000, seed=42, difficulty="easy"):
    """Generate arithmetic problems with chain-of-thought reasoning.

    Each problem includes a natural-language prompt, a step-by-step reasoning
    chain, and a final numeric answer.  The format follows GSM8K-style
    multi-step arithmetic.

    Parameters
    ----------
    num_samples : int
        Number of problems to generate.
    seed : int
        Random seed for reproducibility.
    difficulty : {"easy", "medium", "hard"}
        Controls the number of operations per problem.

    Returns
    -------
    list[dict]
        Each dict has keys ``prompt``, ``reasoning``, and ``answer``.
    """
    random.seed(seed)
    problems: list[dict[str, Any]] = []

    for _ in range(num_samples):
        if difficulty == "easy":
            # 2-step problem: (a op b) op c
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            c = random.randint(1, 100)
            op1 = random.choice(["+", "-", "*"])
            op2 = random.choice(["+", "-", "*"])
            expr1 = f"{a} {op1} {b}"
            val1, err = _safe_eval(expr1)
            if err or val1 is None:
                continue
            expr2 = f"{val1} {op2} {c}"
            val2, err = _safe_eval(expr2)
            if err or val2 is None:
                continue
            reasoning = f"Step 1: {expr1} = {val1}\nStep 2: {expr2} = {val2}"
            prompt = f"Calculate {a} {op1} {b} {op2} {c}. Show your reasoning."
            answer = str(val2)
        elif difficulty == "medium":
            # 3-step with parentheses
            a = random.randint(1, 50)
            b = random.randint(1, 50)
            c = random.randint(1, 50)
            d = random.randint(1, 50)
            op1 = random.choice(["+", "-", "*", "/"])
            op2 = random.choice(["+", "-", "*"])
            expr1 = f"({a} {op1} {b}) {op2} {c}"
            val1, err = _safe_eval(expr1)
            if err or val1 is None:
                continue
            expr2 = f"{val1} + {d}"
            val2, err = _safe_eval(expr2)
            if err or val2 is None:
                continue
            reasoning = f"Step 1: {expr1} = {val1}\nStep 2: {val1} + {d} = {val2}"
            prompt = f"Calculate ({a} {op1} {b}) {op2} {c} + {d}. Show your reasoning."
            answer = str(val2)
        else:
            # hard: nested expressions
            a = random.randint(1, 20)
            b = random.randint(1, 20)
            c = random.randint(1, 20)
            d = random.randint(1, 20)
            expr1 = f"({a} + {b}) * {c}"
            val1, err = _safe_eval(expr1)
            if err or val1 is None:
                continue
            expr2 = f"{val1} / {d}"
            val2, err = _safe_eval(expr2)
            if err or val2 is None:
                continue
            reasoning = f"Step 1: {expr1} = {val1}\nStep 2: {val1} / {d} = {val2}"
            prompt = f"Calculate ({a} + {b}) * {c} / {d}. Show your reasoning."
            answer = str(val2)

        problems.append(
            {
                "prompt": prompt,
                "reasoning": reasoning,
                "answer": answer,
            }
        )

    return problems


class ChainOfThoughtDataset(Dataset):
    """Arithmetic dataset with chain-of-thought reasoning tokens.

    Each sample is formatted as::

        <reasoning>step1
        step2
        ...</reasoning><answer>final_answer</answer>

    Parameters
    ----------
    data : list[dict]
        Output of ``generate_cot_problems``.
    tokenizer : tiktoken.Encoding
    max_length : int
    pre_tokenize : bool
        If True (default), encode all samples in ``__init__``.
    """

    def __init__(self, data, tokenizer, max_length=512, pre_tokenize=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pre_tokenize = pre_tokenize

        if pre_tokenize:
            self.samples = [self._encode_item(item) for item in data]
        else:
            self.samples = None

    def _encode_item(self, item):
        reasoning = item["reasoning"]
        answer = item["answer"]
        full_text = f"<reasoning>{reasoning}</reasoning><answer>{answer}</answer>"
        tokens = self.tokenizer.encode(full_text)
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "reasoning": reasoning,
            "answer": answer,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.samples is not None:
            return self.samples[idx]
        return self._encode_item(self.data[idx])


def collate_fn(batch, pad_token_id=50256):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = []
    labels = []
    for b in batch:
        inp = b["input_ids"]
        lab = b["labels"]
        pad_len = max_len - len(inp)
        if pad_len > 0:
            inp = torch.cat(
                [inp, torch.full((pad_len,), pad_token_id, dtype=torch.long)]
            )
            lab = torch.cat([lab, torch.full((pad_len,), -1, dtype=torch.long)])
        input_ids.append(inp)
        labels.append(lab)
    result = {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
    }
    if "prompt" in batch[0]:
        result["prompts"] = [b["prompt"] for b in batch]
    if "answer" in batch[0]:
        result["answers"] = [b["answer"] for b in batch]
    if "reasoning" in batch[0]:
        result["reasonings"] = [b["reasoning"] for b in batch]
    return result


# ---------------------------------------------------------------------------
# Unit-test stubs
# ---------------------------------------------------------------------------
def _test_cot_generation():
    data = generate_cot_problems(num_samples=5, seed=42, difficulty="easy")
    assert len(data) == 5
    assert all(
        "prompt" in item and "reasoning" in item and "answer" in item for item in data
    )
    print("generate_cot_problems smoke test passed")


def _test_chain_of_thought_dataset():
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    data = generate_cot_problems(num_samples=3, seed=42, difficulty="easy")
    ds = ChainOfThoughtDataset(data, tokenizer, max_length=256)
    assert len(ds) == 3
    sample = ds[0]
    assert "input_ids" in sample
    assert "reasoning" in sample
    print("ChainOfThoughtDataset smoke test passed")


if __name__ == "__main__":
    _test_cot_generation()
    _test_chain_of_thought_dataset()
