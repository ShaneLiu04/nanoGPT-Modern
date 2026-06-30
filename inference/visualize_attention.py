"""Attention visualization and interpretability tools for nanoGPT-Modern.

Provides:
* ``AttentionVisualizer`` — export per-layer attention heatmaps as PNG or
  interactive HTML.
* ``LogitLens`` — inspect the top-k vocabulary predictions at every layer
  during generation, useful for debugging "where the model changes its mind".

Dependencies
------------
matplotlib, seaborn, and (optional) jinja2 for HTML rendering.  These are
soft-imported so that the module can be loaded in headless environments without
GUI libraries.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionVisualizer:
    """Capture and render attention weight maps for a ModernGPT forward pass.

    The visualizer registers a forward hook on every ``CausalSelfAttention``
    layer in the model, records the attention matrix (or the post-softmax
    attention scores if available), and produces heatmaps.

    Parameters
    ----------
    model : nn.Module
        A ``ModernGPT`` instance (or any model with ``transformer.h`` blocks).
    tokenizer : Any
        Tokenizer with ``decode`` or ``convert_ids_to_tokens`` method.
    """

    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self._hooks: List[Any] = []
        self._attn_maps: Dict[int, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on each attention layer."""
        blocks = getattr(self.model.transformer, "h", None)
        if blocks is None:
            raise ValueError("Model does not have transformer.h blocks")
        for idx, block in enumerate(blocks):
            attn = getattr(block, "attn", None)
            if attn is None:
                continue

            # Hook on the attention module to capture the attention matrix.
            def make_hook(layer_idx):
                def hook(mod, inp, out):
                    # Attempt to capture the attention weights if the module
                    # stores them (e.g. via a custom attribute or if we compute
                    # them manually in eager mode).  For SDPA backends the weights
                    # are not returned, so we fall back to computing Q@K^T.
                    self._attn_maps[layer_idx] = self._extract_attention(mod, inp, out)

                return hook

            h = attn.register_forward_hook(make_hook(idx))
            self._hooks.append(h)

    def _extract_attention(
        self, mod: nn.Module, inp: Tuple, out: Tuple
    ) -> torch.Tensor:
        """Extract attention scores from a CausalSelfAttention forward call.

        This is a best-effort heuristic: when the module uses the eager/manual
        path (no SDPA), we can recompute the scores from Q and K.  When SDPA is
        used, the attention weights are not returned, so we synthesise a
        placeholder from the Q/K projections if available.
        """
        # inp is a tuple of (x, past_kv, use_cache, start_pos, ...)
        x = inp[0] if isinstance(inp, tuple) else inp
        B, T, C = x.shape
        # Recompute Q and K to synthesise attention scores.
        q = mod.q_proj(x).view(B, T, mod.n_head, mod.head_dim).transpose(1, 2)
        k = mod.k_proj(x).view(B, T, mod.n_kv_head, mod.head_dim).transpose(1, 2)
        # RoPE is applied in the forward; for viz we approximate raw scores.
        # Expand K to match Q heads for visualization.
        if mod.n_kv_head < mod.n_head:
            n_rep = mod.n_head // mod.n_kv_head
            k = k.repeat_interleave(n_rep, dim=1)
        scores = (q @ k.transpose(-2, -1)) / (mod.head_dim**0.5)
        # Causal mask.
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        attn = F.softmax(scores, dim=-1)
        return attn.detach().cpu()  # [B, n_head, T, T]

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def plot_heatmaps(
        self,
        tokens: List[str],
        layer_idx: Optional[int] = None,
        head_idx: Optional[int] = None,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> Optional[Any]:
        """Render attention heatmaps with matplotlib / seaborn.

        Parameters
        ----------
        tokens : list[str]
            Token strings for axis labels.
        layer_idx : int or None
            If set, plot only this layer; otherwise plot all layers in subplots.
        head_idx : int or None
            If set, plot only this head; otherwise average over heads.
        output_path : str or None
            If set, save the figure to this path (PNG).
        figsize : tuple
            Matplotlib figure size.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError as e:
            raise ImportError(
                "AttentionVisualizer.plot_heatmaps requires matplotlib and seaborn"
            ) from e

        if not self._attn_maps:
            raise RuntimeError("No attention maps captured. Run a forward pass first.")

        layers = (
            [layer_idx] if layer_idx is not None else sorted(self._attn_maps.keys())
        )
        n_layers = len(layers)
        cols = min(4, n_layers)
        rows = math.ceil(n_layers / cols)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_layers == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for ax, li in zip(axes, layers):
            attn = self._attn_maps[li]  # [B, n_head, T, T]
            if head_idx is not None:
                mat = attn[0, head_idx]  # [T, T]
                title = f"Layer {li} Head {head_idx}"
            else:
                mat = attn[0].mean(dim=0)  # [T, T]
                title = f"Layer {li} (avg over {attn.shape[1]} heads)"

            # Truncate to token length.
            T = min(mat.shape[0], len(tokens))
            mat_np = mat[:T, :T].numpy()
            sns.heatmap(
                mat_np,
                ax=ax,
                cmap="YlOrRd",
                xticklabels=tokens[:T],
                yticklabels=tokens[:T],
            )
            ax.set_title(title)
            ax.tick_params(axis="both", labelsize=6)

        # Hide unused subplots.
        for ax in axes[n_layers:]:
            ax.axis("off")

        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
        return fig

    def to_html(
        self,
        tokens: List[str],
        output_path: str,
        layer_indices: Optional[List[int]] = None,
    ) -> str:
        """Export attention heatmaps as an interactive HTML file.

        Uses a simple JavaScript + Canvas heatmap renderer so that no extra
        Python GUI libraries are required at runtime.
        """
        if not self._attn_maps:
            raise RuntimeError("No attention maps captured. Run a forward pass first.")

        layers = (
            layer_indices
            if layer_indices is not None
            else sorted(self._attn_maps.keys())
        )
        data = {}
        for li in layers:
            attn = self._attn_maps[li][0].mean(dim=0).numpy().tolist()  # [T, T]
            data[f"layer_{li}"] = attn

        html = self._render_html(tokens, data)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path

    @staticmethod
    def _render_html(tokens: List[str], data: Dict[str, List[List[float]]]) -> str:
        """Build an HTML page with a selectable heatmap viewer."""
        token_json = json.dumps(tokens)
        data_json = json.dumps(data)
        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Attention Heatmap</title>
<style>
  body {{ font-family: sans-serif; margin: 20px; }}
  #controls {{ margin-bottom: 10px; }}
  canvas {{ border: 1px solid #ccc; image-rendering: pixelated; }}
  #tooltip {{ position: absolute; background: rgba(0,0,0,0.7); color: white;
             padding: 4px 8px; border-radius: 4px; font-size: 12px; pointer-events: none; display: none; }}
</style>
</head>
<body>
<h2>Attention Visualizer</h2>
<div id="controls">
  <label>Layer: <select id="layerSelect"></select></label>
  <span id="info"></span>
</div>
<div style="position:relative;">
  <canvas id="heatmap"></canvas>
  <div id="tooltip"></div>
</div>
<script>
const tokens = {token_json};
const attnData = {data_json};
const layerSelect = document.getElementById('layerSelect');
Object.keys(attnData).forEach(k => {{
  const opt = document.createElement('option');
  opt.value = k; opt.textContent = k;
  layerSelect.appendChild(opt);
}});
const canvas = document.getElementById('heatmap');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
const cellSize = 16;
function draw(layerKey) {{
  const mat = attnData[layerKey];
  const T = mat.length;
  canvas.width = T * cellSize;
  canvas.height = T * cellSize;
  const maxVal = Math.max(...mat.flat());
  for (let i = 0; i < T; i++) {{
    for (let j = 0; j < T; j++) {{
      const v = mat[i][j] / maxVal;
      const r = Math.floor(255 * v);
      const g = Math.floor(255 * (1 - v));
      ctx.fillStyle = `rgb(${{r}},${{g}},0)`;
      ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
    }}
  }}
}}
layerSelect.addEventListener('change', () => draw(layerSelect.value));
canvas.addEventListener('mousemove', e => {{
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((e.clientX - rect.left) / cellSize);
  const y = Math.floor((e.clientY - rect.top) / cellSize);
  const mat = attnData[layerSelect.value];
  if (y >= 0 && y < mat.length && x >= 0 && x < mat.length) {{
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 10) + 'px';
    tooltip.style.top = (e.clientY + 10) + 'px';
    tooltip.textContent = `${{tokens[y]}} -> ${{tokens[x]}} : ${{mat[y][x].toFixed(4)}}`;
  }} else {{
    tooltip.style.display = 'none';
  }}
}});
draw(layerSelect.value);
</script>
</body>
</html>"""


class LogitLens:
    """Logit Lens: inspect top-k vocabulary predictions at every layer.

    During generation, the Logit Lens projects the hidden state at each layer
    through the unembedding matrix and reports the top-k tokens.  This reveals
    "when" the model has decided on the next token — useful for interpretability
    debugging.

    Parameters
    ----------
    model : nn.Module
    tokenizer : Any
    top_k : int
        Number of top tokens to report per layer.
    """

    def __init__(self, model: nn.Module, tokenizer: Any, top_k: int = 5):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self._lm_head = model.lm_head
        self._hooks: List[Any] = []
        self._records: List[Dict[str, Any]] = []

    def _register_hooks(self) -> None:
        """Register hooks on every transformer block to capture hidden states."""
        blocks = getattr(self.model.transformer, "h", None)
        if blocks is None:
            raise ValueError("Model does not have transformer.h blocks")
        for idx, block in enumerate(blocks):

            def make_hook(layer_idx):
                def hook(mod, inp, out):
                    # out is (x, present_kv) or x depending on the path.
                    x = out[0] if isinstance(out, tuple) else out
                    self._record_layer(x, layer_idx)

                return hook

            h = block.register_forward_hook(make_hook(idx))
            self._hooks.append(h)

    def _record_layer(self, hidden: torch.Tensor, layer_idx: int) -> None:
        """Project hidden state to vocab and store top-k tokens."""
        # hidden: [B, T, C]
        with torch.no_grad():
            logits = self._lm_head(hidden)  # [B, T, V]
            probs = F.softmax(logits, dim=-1)
            # Focus on the last token position.
            last_logits = logits[:, -1, :]  # [B, V]
            top_vals, top_ids = torch.topk(last_logits, self.top_k, dim=-1)
            top_probs = probs[:, -1, :].gather(-1, top_ids)
            self._records.append(
                {
                    "layer": layer_idx,
                    "tokens": top_ids.cpu().tolist(),
                    "logits": top_vals.cpu().tolist(),
                    "probs": top_probs.cpu().tolist(),
                }
            )

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def attach(self) -> None:
        """Start recording layer-wise predictions."""
        self._records.clear()
        self._register_hooks()

    def detach(self) -> None:
        """Stop recording and remove hooks."""
        self.remove_hooks()

    def get_records(self) -> List[Dict[str, Any]]:
        """Return the captured records from the last forward pass."""
        return self._records

    def decode_records(self, skip_special: bool = True) -> List[Dict[str, Any]]:
        """Decode token ids to strings using the tokenizer.

        Returns a list of dicts with keys ``layer``, ``tokens``, ``texts``,
        ``probs``.
        """
        decoded = []
        for rec in self._records:
            ids = rec["tokens"][0]  # first batch item
            texts = []
            for tid in ids:
                try:
                    t = self.tokenizer.decode([tid], skip_special_tokens=skip_special)
                except Exception:
                    t = str(tid)
                texts.append(t)
            decoded.append(
                {
                    "layer": rec["layer"],
                    "tokens": ids,
                    "texts": texts,
                    "probs": rec["probs"][0],
                }
            )
        return decoded

    def print_during_generation(self, prompt: str, max_new_tokens: int = 10) -> None:
        """Convenience wrapper: generate and print Logit Lens output per step."""
        print(f"LogitLens over {max_new_tokens} steps for prompt: {prompt!r}")
        # Encode prompt.
        try:
            import tiktoken

            enc = tiktoken.get_encoding("gpt2")
            ids = enc.encode(prompt)
        except Exception:
            ids = [ord(c) for c in prompt]
        idx = torch.tensor(
            [ids], dtype=torch.long, device=next(self.model.parameters()).device
        )

        self.model.eval()
        with torch.no_grad():
            for step in range(max_new_tokens):
                self._records.clear()
                self.attach()
                logits, _, _ = self.model(idx)
                self.detach()
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_id], dim=1)
                decoded = self.decode_records()
                print(f"\n--- Step {step + 1} (next token id={next_id.item()}) ---")
                for rec in decoded:
                    top = ", ".join(
                        f"{t!r} ({p:.3f})" for t, p in zip(rec["texts"], rec["probs"])
                    )
                    print(f"  Layer {rec['layer']:2d}: {top}")


import math  # noqa: E402, F811
