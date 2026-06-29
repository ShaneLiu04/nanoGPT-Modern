"""Batched, variable-length generation helpers.

These utilities right-pad prompt batches and use ``attention_mask`` so prompts
of different lengths can be generated together without explicit left-padding
position bookkeeping.  The no-cache path is used to avoid KV-cache complexity
with padding; it is intended for short prompts (e.g. arithmetic evaluation and
GRPO rollouts).
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sample_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """Sample a single token from logits [B, vocab]."""
    if temperature is not None and temperature > 0 and temperature != 1.0:
        logits = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k, dim=-1)
        logits[logits < v[:, [-1]]] = -float("Inf")
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cum_probs > top_p
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False
        sorted_logits[sorted_mask] = -float("Inf")
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def batched_generate(
    model: nn.Module,
    prompts: List[str],
    tokenizer,
    max_new_tokens: int,
    device: Union[str, torch.device],
    batch_size: int = 16,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    return_ids: bool = False,
):
    """
    Generate responses for a list of prompt strings in variable-length batches.

    Parameters
    ----------
    model : nn.Module
        Model with a ``forward(input_ids, attention_mask=...) -> logits`` API.
    prompts : list[str]
    tokenizer : tiktoken.Encoding
    max_new_tokens : int
    device : str
    batch_size : int
    temperature, top_k, top_p : generation hyperparameters
    eos_token_id : int or None
        Defaults to tokenizer's EOT token.
    return_ids : bool
        If True, also return a list of response token-id lists.

    Returns
    -------
    responses : list[str]
    response_ids_list : list[list[int]] (only if return_ids=True)
    """
    if eos_token_id is None:
        eos_token_id = tokenizer.eot_token

    prompt_tokens = [tokenizer.encode(p) for p in prompts]
    prompt_lens = [len(t) for t in prompt_tokens]

    all_responses: List[str] = []
    all_response_ids: List[List[int]] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(prompt_tokens), batch_size):
            batch_prompts = prompt_tokens[start:start + batch_size]
            batch_lens = prompt_lens[start:start + batch_size]
            B = len(batch_prompts)

            sequences = [list(toks) for toks in batch_prompts]
            finished = [False] * B

            for _ in range(max_new_tokens):
                if all(finished):
                    break

                max_len = max(len(s) for s in sequences)
                input_ids = torch.full((B, max_len), eos_token_id, dtype=torch.long, device=device)
                attention_mask = torch.zeros((B, max_len), dtype=torch.bool, device=device)
                for b, seq in enumerate(sequences):
                    input_ids[b, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                    attention_mask[b, :len(seq)] = True

                logits, _, _ = model(input_ids, attention_mask=attention_mask, use_cache=False)
                # Gather the logit at the last real token of each sequence.
                logits_last = torch.stack([logits[b, len(sequences[b]) - 1] for b in range(B)])
                next_tokens = _sample_logits(logits_last, temperature, top_k, top_p)

                for b in range(B):
                    if not finished[b]:
                        tok = next_tokens[b].item()
                        sequences[b].append(tok)
                        if tok == eos_token_id:
                            finished[b] = True

            for seq, plen in zip(sequences, batch_lens):
                resp_ids = seq[plen:]
                if eos_token_id in resp_ids:
                    resp_ids = resp_ids[:resp_ids.index(eos_token_id)]
                all_response_ids.append(resp_ids)
                all_responses.append(tokenizer.decode(resp_ids))

    if return_ids:
        return all_responses, all_response_ids
    return all_responses


def generate_by_length(
    model: nn.Module,
    prompts: List[str],
    tokenizer,
    max_new_tokens: int,
    device: Union[str, torch.device],
    batch_size: int = 16,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    return_ids: bool = False,
):
    """
    Generate responses by grouping prompts of the same length into cache-enabled batches.

    This avoids padding/position issues and uses the model's native ``generate()``
    with KV cache, so it is much faster than per-prompt sequential generation when
    prompts share common lengths.
    """
    if eos_token_id is None:
        eos_token_id = tokenizer.eot_token

    prompt_tokens = [tokenizer.encode(p) for p in prompts]

    by_length = defaultdict(list)
    for i, toks in enumerate(prompt_tokens):
        by_length[len(toks)].append((i, toks))

    response_ids: List[Optional[List[int]]] = [None] * len(prompts)

    model.eval()
    with torch.no_grad():
        for length, items in by_length.items():
            for start in range(0, len(items), batch_size):
                batch_items = items[start:start + batch_size]
                idx = torch.tensor(
                    [toks for _, toks in batch_items],
                    dtype=torch.long,
                    device=device,
                )
                generated = model.generate(
                    idx,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    use_cache=True,
                    eos_token_id=eos_token_id,
                )
                for j, (orig_i, toks) in enumerate(batch_items):
                    resp = generated[j, len(toks):].tolist()
                    if eos_token_id in resp:
                        resp = resp[:resp.index(eos_token_id)]
                    response_ids[orig_i] = resp

    responses = [tokenizer.decode(r) for r in response_ids]
    if return_ids:
        return responses, response_ids
    return responses
