"""
BaselineGPT: Standard GPT-2 architecture with LayerNorm, GELU FFN, and absolute positional embeddings.
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class CausalSelfAttention(nn.Module):
    """Standard GPT-2 causal self-attention with dual-backend support.

    Two backends are selectable via the Config:

    * `"sdpa"` (default) : uses PyTorch `scaled_dot_product_attention`,
      which automatically dispatches to FlashAttention / Memory-Efficient
      Attention when available.  This is the same backend that ModernGPT uses,
      making pre-training control experiments directly comparable.

    * `"manual"` : the classic hand-rolled matmul + causal mask path
      (original nanoGPT).  Useful for pedagogical comparison and for
      verifying that the backend choice does not change the forward result.

    All trainable parameters are **identical** regardless of the backend; only
    the computation graph differs.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.backend = getattr(config, "attention_backend", "sdpa")

        # --- common projection layers ---
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # --- causal mask (only needed by "manual" backend) ---
        if self.backend == "manual":
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )
        else:
            self.register_buffer("bias", torch.empty(0))  # never used

    def _forward_manual(self, q, k, v, T):
        """Original nanoGPT attention path (matmul + mask + softmax)."""
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att @ v

    def _forward_sdpa(self, q, k, v, T):
        """PyTorch native scaled_dot_product_attention (FlashAttention dispatcher)."""
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=True,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )

    def forward(self, x):
        B, T, C = x.size()

        # project Q / K / V from the fused projection matrix
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # reshape: [B, T, n_embd] -> [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # dispatch to the chosen attention backend
        if self.backend == "manual":
            y = self._forward_manual(q, k, v, T)
        else:
            y = self._forward_sdpa(q, k, v, T)

        # output projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with configurable Pre-Norm / Post-Norm.

    * `norm_position="pre"` (default, LLaMA-style):
      `x = x + Attn(Norm(x));  x = x + MLP(Norm(x))`

    * `norm_position="post"` (GPT-2 original):
      `x = Norm(x + Attn(x));  x = Norm(x + MLP(x))`
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm_pos = getattr(config, "norm_position", "pre")
        if self.norm_pos not in ("pre", "post"):
            raise ValueError(f"norm_position must be 'pre' or 'post', got '{self.norm_pos}'")

        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        gc_enabled = (
            self.training
            and getattr(self.config, "gradient_checkpointing", False)
        )

        if self.norm_pos == "pre":
            if gc_enabled:
                x = x + checkpoint(self.attn, self.ln_1(x), use_reentrant=False)
                x = x + checkpoint(self.mlp, self.ln_2(x), use_reentrant=False)
            else:
                x = x + self.attn(self.ln_1(x))
                x = x + self.mlp(self.ln_2(x))
        else:
            if gc_enabled:
                x = self.ln_1(x + checkpoint(self.attn, x, use_reentrant=False))
                x = self.ln_2(x + checkpoint(self.mlp, x, use_reentrant=False))
            else:
                residual = x
                x = self.ln_1(residual + self.attn(x))
                x = self.ln_2(x + self.mlp(x))
        return x


class BaselineGPTConfig:
    """Configuration for BaselineGPT (standard GPT-2).

    Parameters
    ----------
    attention_backend : str
        `"sdpa"` (default) or `"manual"`.  `"sdpa"` uses PyTorch native
        scaled_dot_product_attention (FlashAttention); `"manual"` uses the
        original nanoGPT matmul + causal-mask path.  Both produce identical
        forward results (to within floating-point tolerance) and have the same
        trainable parameters.
    """
    def __init__(
        self,
        vocab_size=50257,
        block_size=1024,
        n_layer=9,
        n_head=8,
        n_embd=512,
        dropout=0.0,
        bias=True,
        attention_backend="sdpa",
        norm_position="pre",
        gradient_checkpointing=False,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        if attention_backend not in ("sdpa", "manual"):
            raise ValueError(
                f"attention_backend must be 'sdpa' or 'manual', got '{attention_backend}'"
            )
        self.attention_backend = attention_backend
        if norm_position not in ("pre", "post"):
            raise ValueError(
                f"norm_position must be 'pre' or 'post', got '{norm_position}'"
            )
        self.norm_position = norm_position
        self.gradient_checkpointing = gradient_checkpointing

    def to_dict(self) -> dict:
        """Serialize the config to a shallow dict for checkpointing."""
        return {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "dropout": self.dropout,
            "bias": self.bias,
            "attention_backend": self.attention_backend,
            "norm_position": self.norm_position,
            "gradient_checkpointing": self.gradient_checkpointing,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BaselineGPTConfig":
        """Deserialize from a dict, ignoring unknown keys."""
        return cls(**{k: v for k, v in d.items() if k in cls.__init__.__code__.co_varnames})


class BaselineGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight.data = self.transformer.wpe.weight.data[:block_size]
        for block in self.transformer.h:
            if hasattr(block.attn, "bias") and block.attn.bias.numel() > 0:
                block.attn.bias.data = block.attn.bias.data[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Deduplicate parameters by tensor id.  wte and lm_head share the same
        # weight tensor, so without deduplication AdamW would update it twice.
        param_dict = {id(p): (n, p) for n, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        import inspect
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
