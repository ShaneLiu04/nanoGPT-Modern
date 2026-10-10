"""Iteration-level continuous batching scheduler for nanoGPT-Modern.

FR7 rewrite: the scheduler advances the whole batch one *iteration* at a
time.  Each iteration harvests FINISHED slots, pops queued requests into
the freed slots, prefills the new-request group (left-aligned, right pads
never enter the KV buffer, per-sequence positions 0..pl-1, first token
sampled from the prompt's last real position) and steps every DECODING
slot once, sampling all tokens of the iteration through a single
vectorized call.  A request that finishes therefore frees its slot at the
very next iteration boundary instead of waiting for the rest of the batch.

The former placeholder prefix-cache class and the never-called
decode-step compile helper were removed (dead code); ``RequestQueue`` is
protected by a ``threading.Lock``.

KV ownership: the scheduler allocates one dense buffer per layer of shape
[batch_size, n_kv_head, max_cache_len, head_dim].  Each slot row holds
only its own real tokens (``slot.cache_len``); rows never share state, so
an empty slot cannot contaminate its neighbours (B12).  Decode steps pass
each row's valid prefix via ``kv_mask`` and the new token's absolute
position via ``positions`` (see ``ModernGPT.forward``).

Usage Example
-------------
>>> queue = RequestQueue()
>>> queue.enqueue(Request(prompt_ids=[1, 2, 3], max_tokens=16, request_id="a"))
>>> scheduler = ContinuousBatchScheduler(queue, model, device="cpu", batch_size=4)
>>> results = scheduler.run()          # List[GenerateResult]
"""

from __future__ import annotations

import dataclasses
import enum
import threading
from typing import List, Optional, Union

import torch
import torch.nn as nn

from inference.generate_utils import _sample_logits_batched


@dataclasses.dataclass
class Request:
    """Single inference request.

    Attributes
    ----------
    prompt_ids : torch.Tensor [1, T] or list[int]
        Prompt token ids.  Internally normalized to a [1, T] long tensor.
    max_tokens : int
        Maximum number of *new* tokens to generate.
    temperature : float
        Sampling temperature; ``0.0`` means greedy argmax.
    top_k : int or None
        Top-k filtering (None or <= 0 -> no filtering).
    top_p : float or None
        Nucleus sampling threshold (None or >= 1.0 -> no filtering).
    repetition_penalty : float or None
        Divide the logits of already-seen token ids by this factor
        (None or 1.0 -> disabled).
    eos_token_id : int or None
        Sampling this token finishes the request early (None -> never
        stop on EOS).
    request_id : str
        Unique identifier returned with the result.
    """

    prompt_ids: Union[torch.Tensor, List[int]]
    max_tokens: int = 128
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    eos_token_id: Optional[int] = None
    request_id: str = ""

    def __post_init__(self):
        if isinstance(self.prompt_ids, list):
            self.prompt_ids = torch.tensor([self.prompt_ids], dtype=torch.long)
        elif self.prompt_ids.dim() == 1:
            self.prompt_ids = self.prompt_ids.unsqueeze(0)


class RequestQueue:
    """Thread-safe in-process request queue for continuous batching.

    ``enqueue`` raises ``RuntimeError`` when the queue is full (it does
    not block); all operations are protected by a ``threading.Lock``.
    """

    def __init__(self, maxsize: int = 1000):
        self._queue: List[Request] = []
        self.maxsize = maxsize
        self._lock = threading.Lock()

    def enqueue(self, req: Request) -> None:
        """Add a request to the tail of the queue."""
        with self._lock:
            if len(self._queue) >= self.maxsize:
                raise RuntimeError("Request queue is full")
            self._queue.append(req)

    def dequeue(self) -> Optional[Request]:
        """Pop the oldest request, or ``None`` if the queue is empty."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue.pop(0)

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)

    def is_empty(self) -> bool:
        return len(self) == 0


@dataclasses.dataclass
class GenerateResult:
    """A completed request, as returned by ``ContinuousBatchScheduler.run``.

    Attributes
    ----------
    request_id : str
    token_ids : list[int]
        Full generated sequence: prompt + completion (the sampled EOS
        token, if any, is included at the end).
    finish_reason : str
        ``"eos"`` when the request stopped on its eos token, ``"length"``
        when it ran to ``max_tokens``.
    finish_iteration : int
        1-based index of the iteration in which the request finished.
    """

    request_id: str
    token_ids: List[int]
    finish_reason: str
    finish_iteration: int


class _SlotPhase(enum.Enum):
    EMPTY = 0
    PREFILL = 1  # accepted from the queue, waiting for its prefill forward
    DECODING = 2
    FINISHED = 3


@dataclasses.dataclass
class _Slot:
    """Per-slot state of the iteration-level scheduler."""

    request: Request
    prompt_len: int
    completion: List[int]
    cache_len: int  # number of real KV entries in this slot's buffer row
    phase: _SlotPhase = _SlotPhase.PREFILL
    finish_reason: str = ""
    finish_iteration: int = 0


class ContinuousBatchScheduler:
    """Iteration-level scheduler driven by a request queue.

    Each iteration: harvest FINISHED slots -> pop new requests into freed
    slots -> prefill the new-request group (left-aligned) -> step every
    DECODING slot one token -> vectorized sampling.  ``run`` returns the
    results of all requests that finished within the step budget.

    Parameters
    ----------
    queue : RequestQueue
    model : nn.Module
        A ``ModernGPT`` (needs ``.config`` with n_layer/n_head/n_kv_head/
        n_embd/effective_block_size).
    device : str or torch.device
    batch_size : int
        Maximum number of concurrent sequences (KV buffer rows).
    """

    def __init__(
        self,
        queue: RequestQueue,
        model: nn.Module,
        device: Union[str, torch.device] = "cpu",
        batch_size: int = 4,
    ):
        self.queue = queue
        self.model = model
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.iteration_count = 0
        cfg = model.config
        self._n_layer = cfg.n_layer
        self._n_kv_head = cfg.n_kv_head
        self._head_dim = cfg.n_embd // cfg.n_head
        self._max_cache_len = cfg.effective_block_size
        self._k_buf: Optional[List[torch.Tensor]] = None
        self._v_buf: Optional[List[torch.Tensor]] = None
        self._slots: List[Optional[_Slot]] = [None] * batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, max_steps: Optional[int] = None) -> List[GenerateResult]:
        """Run the scheduler until the queue is empty and every slot has
        finished, or until ``max_steps`` iterations have been executed.

        Parameters
        ----------
        max_steps : int or None
            Iteration budget.  ``None`` runs to completion.  Requests
            that have not finished within the budget do not produce a
            ``GenerateResult``.

        Returns
        -------
        list[GenerateResult]
            Results of all finished requests, in completion order.
        """
        self.iteration_count = 0
        self._slots = []
        self._ensure_buffers()
        results: List[GenerateResult] = []

        while True:
            # 1. Harvest FINISHED slots into results and compact the
            #    buffer so active slots always occupy rows [0..K): the
            #    decode forward can then slice its past as a zero-copy
            #    view instead of gathering rows every step.
            keep: List[int] = []
            for i, slot in enumerate(self._slots):
                if slot is None:
                    continue
                if slot.phase == _SlotPhase.FINISHED:
                    results.append(
                        GenerateResult(
                            request_id=slot.request.request_id,
                            token_ids=slot.request.prompt_ids[0].tolist()
                            + slot.completion,
                            finish_reason=slot.finish_reason,
                            finish_iteration=slot.finish_iteration,
                        )
                    )
                else:
                    keep.append(i)
            if len(keep) != len(self._slots):
                if keep != list(range(len(keep))):
                    idx = torch.tensor(keep, dtype=torch.long, device=self.device)
                    k_buf = self._k_buf
                    v_buf = self._v_buf
                    assert k_buf is not None and v_buf is not None
                    for li in range(self._n_layer):
                        # Fancy indexing copies, so writing the prefix
                        # back in place is aliasing-safe.  The buffer keeps
                        # its full batch_size row count.
                        k_buf[li][: len(keep)] = k_buf[li][idx]
                        v_buf[li][: len(keep)] = v_buf[li][idx]
                self._slots = [self._slots[i] for i in keep]

            # 2. Back-fill empty slots at the iteration boundary (new
            #    requests append at the tail rows).
            while len(self._slots) < self.batch_size and not self.queue.is_empty():
                req = self.queue.dequeue()
                if req is None:
                    break
                needed = req.prompt_ids.shape[1] + req.max_tokens
                if needed > self._max_cache_len:
                    raise ValueError(
                        f"request {req.request_id!r} needs "
                        f"{needed} cache slots (prompt + "
                        f"max_tokens) but the effective block "
                        f"size is {self._max_cache_len}"
                    )
                self._slots.append(
                    _Slot(
                        request=req,
                        prompt_len=req.prompt_ids.shape[1],
                        completion=[],
                        cache_len=0,
                    )
                )

            # 3. Budget / termination checks.
            if max_steps is not None and self.iteration_count >= max_steps:
                break
            if not self._slots:
                break

            self.iteration_count += 1

            # 4. Snapshot the decode group first: slots accepted in this
            #    iteration prefill below and join the decode group only in
            #    the next iteration (their first token comes from prefill).
            #    DECODING slots are exactly the contiguous rows [0..K).
            n_decoding = sum(
                1 for s in self._slots if s.phase == _SlotPhase.DECODING
            )

            # 5. Prefill group (slots accepted since the last boundary).
            prefill_rows = [
                i
                for i, s in enumerate(self._slots)
                if s.phase == _SlotPhase.PREFILL
            ]
            if prefill_rows:
                self._prefill(prefill_rows)

            # 6. Decode: every slot that was DECODING at the iteration
            #    boundary advances one token.
            if n_decoding:
                self._decode(list(range(n_decoding)))

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_buffers(self) -> None:
        if self._k_buf is not None:
            return
        dtype = next(self.model.parameters()).dtype
        shape = (
            self.batch_size,
            self._n_kv_head,
            self._max_cache_len,
            self._head_dim,
        )
        self._k_buf = [
            torch.zeros(shape, dtype=dtype, device=self.device)
            for _ in range(self._n_layer)
        ]
        self._v_buf = [
            torch.zeros(shape, dtype=dtype, device=self.device)
            for _ in range(self._n_layer)
        ]

    def _prefill(self, rows: List[int]) -> None:
        """Left-aligned prefill of the new-request group.

        Prompts are packed from position 0 (so implicit RoPE/wpe positions
        0..pl-1 are correct for every row); right pads are masked out and
        their K/V are *never* written into the buffer.  The first token is
        sampled from each row's last real position, fixing the legacy
        off-by-one where the prompt tail was re-fed on the first decode
        step.
        """
        slots = [self._slots[i] for i in rows]
        P = len(slots)
        max_pl = max(s.prompt_len for s in slots)
        input_ids = torch.zeros((P, max_pl), dtype=torch.long, device=self.device)
        attn = torch.zeros((P, max_pl), dtype=torch.bool, device=self.device)
        for j, s in enumerate(slots):
            pl = s.prompt_len
            pids = s.request.prompt_ids
            # __post_init__ normalises prompt_ids to a [1, T] tensor
            assert isinstance(pids, torch.Tensor)
            input_ids[j, :pl] = pids[0, :pl].to(self.device)
            attn[j, :pl] = True

        with torch.no_grad():
            logits, _, raw_kvs = self.model(
                input_ids, attention_mask=attn, use_cache=True, start_pos=0
            )

        # Write only each row's real prefix: pad positions never enter KV.
        k_buf = self._k_buf
        v_buf = self._v_buf
        assert k_buf is not None and v_buf is not None
        for li in range(self._n_layer):
            k_new, v_new = raw_kvs[li]
            for j, i in enumerate(rows):
                pl = slots[j].prompt_len
                k_buf[li][i, :, :pl] = k_new[j, :, :pl].to(k_buf[li].dtype)
                v_buf[li][i, :, :pl] = v_new[j, :, :pl].to(v_buf[li].dtype)

        for i in rows:
            s = self._slots[i]
            s.cache_len = s.prompt_len
            s.phase = _SlotPhase.DECODING

        last_logits = torch.stack(
            [logits[j, slots[j].prompt_len - 1] for j in range(P)]
        )
        self._sample_and_finish(rows, last_logits)

    def _decode(self, rows: List[int]) -> None:
        """Advance every DECODING slot in ``rows`` by one token.

        ``rows`` is the contiguous prefix [0..K) of the compact buffer
        layout, so the past is a zero-copy view.  Each row feeds its last
        generated token at its own absolute position (``positions``); the
        shared past carries the per-row valid prefix (``kv_mask``), so
        rows with unequal cache lengths decode correctly in one batched
        forward.
        """
        slots = [self._slots[i] for i in rows]
        D = len(rows)
        lens_t = torch.tensor(
            [s.cache_len for s in slots], dtype=torch.long, device=self.device
        )
        max_len = int(lens_t.max().item())

        next_in = torch.tensor(
            [[s.completion[-1]] for s in slots], dtype=torch.long, device=self.device
        )
        positions = lens_t.unsqueeze(1)  # [D, 1] absolute position per row
        kv_mask = (
            torch.arange(max_len, device=self.device).unsqueeze(0)
            < lens_t.unsqueeze(1)
        )  # [D, max_len] bool: True at each row's real prefix

        k_buf = self._k_buf
        v_buf = self._v_buf
        assert k_buf is not None and v_buf is not None
        past = [
            (
                k_buf[li][:D, :, :max_len, :],
                v_buf[li][:D, :, :max_len, :],
            )
            for li in range(self._n_layer)
        ]

        with torch.no_grad():
            logits, _, new_kvs = self.model(
                next_in,
                past_kvs=past,
                use_cache=True,
                positions=positions,
                kv_mask=kv_mask,
            )

        # Write each row's new K/V at that row's own cache_len: one
        # vectorized indexed assignment per layer (no per-row loop).
        for li in range(self._n_layer):
            k_new, v_new = new_kvs[li]  # [D, n_kv_head, 1, head_dim]
            k_buf[li][
                torch.arange(D, device=self.device), :, lens_t
            ] = k_new[:, :, 0]
            v_buf[li][
                torch.arange(D, device=self.device), :, lens_t
            ] = v_new[:, :, 0]
        for i in rows:
            self._slots[i].cache_len += 1

        self._sample_and_finish(rows, logits[:, -1, :])

    def _sample_and_finish(self, rows: List[int], logits: torch.Tensor) -> None:
        """Vectorized sampling + per-slot finish bookkeeping.

        ``logits`` is [len(rows), V]: one row per slot, gathered at each
        slot's own last real position.  Temperature/top-k/top-p travel as
        [B] tensors through ``_sample_logits_batched`` (no per-request
        Python sampling loop).
        """
        slots = [self._slots[i] for i in rows]
        logits = self._apply_repetition_penalty(logits, slots)

        temps = torch.tensor(
            [s.request.temperature for s in slots],
            dtype=torch.float32,
            device=self.device,
        )
        top_ks = torch.tensor(
            [
                s.request.top_k if s.request.top_k is not None else 0
                for s in slots
            ],
            dtype=torch.long,
            device=self.device,
        )
        top_ps = torch.tensor(
            [
                s.request.top_p if s.request.top_p is not None else 1.0
                for s in slots
            ],
            dtype=torch.float32,
            device=self.device,
        )
        tokens = _sample_logits_batched(logits, temps, top_ks=top_ks, top_ps=top_ps)
        toks = tokens.tolist()

        for j, i in enumerate(rows):
            s = self._slots[i]
            tok = int(toks[j])
            s.completion.append(tok)
            eos = s.request.eos_token_id
            if eos is not None and tok == eos:
                s.phase = _SlotPhase.FINISHED
                s.finish_reason = "eos"
                s.finish_iteration = self.iteration_count
            elif len(s.completion) >= s.request.max_tokens:
                s.phase = _SlotPhase.FINISHED
                s.finish_reason = "length"
                s.finish_iteration = self.iteration_count

    def _apply_repetition_penalty(
        self, logits: torch.Tensor, slots: List[_Slot]
    ) -> torch.Tensor:
        """Divide the logits of each slot's seen token ids by that slot's
        penalty factor.  Vectorized gather/scatter over a padded [B, L]
        index tensor; slots without a penalty pass through unchanged."""
        pens = [s.request.repetition_penalty for s in slots]
        if all(p is None or p == 1.0 for p in pens):
            return logits

        seqs = [
            s.request.prompt_ids[0].tolist() + s.completion for s in slots
        ]
        maxlen = max(len(q) for q in seqs)
        B = len(slots)
        idx = torch.zeros((B, maxlen), dtype=torch.long, device=self.device)
        valid = torch.zeros((B, maxlen), dtype=torch.bool, device=self.device)
        for j, q in enumerate(seqs):
            idx[j, : len(q)] = torch.as_tensor(q, dtype=torch.long, device=self.device)
            valid[j, : len(q)] = True
        pen = torch.tensor(
            [p if p is not None and p != 1.0 else 1.0 for p in pens],
            dtype=logits.dtype,
            device=self.device,
        ).unsqueeze(1)
        gathered = logits.gather(1, idx)
        scaled = torch.where(valid, gathered / pen, gathered)
        return logits.scatter(1, idx, scaled)


class ContinuousBatchingServer:
    """High-level wrapper exposing a simple ``submit``/``generate_pending``
    API for local/offline serving (for HTTP serving see ``api_server.py``).
    """

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = "cpu",
        batch_size: int = 4,
    ):
        self.model = model
        self.queue = RequestQueue()
        self.scheduler = ContinuousBatchScheduler(
            self.queue, model, device=device, batch_size=batch_size
        )

    def submit(self, req: Request) -> None:
        """Enqueue a new request."""
        self.queue.enqueue(req)

    def generate_pending(self, max_steps: Optional[int] = None) -> List[GenerateResult]:
        """Run the scheduler on all pending requests and return results."""
        return self.scheduler.run(max_steps=max_steps)
