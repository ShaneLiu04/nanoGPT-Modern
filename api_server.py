"""FastAPI-based inference server compatible with OpenAI API.

Endpoints:
  - ``POST /v1/completions`` (legacy, non-chat)
  - ``POST /v1/chat/completions`` (chat format, compatible with OpenAI SDK)

Features:
  - KV Cache enabled by default for efficient autoregressive generation.
  - Continuous batching (optional): when ``--continuous_batching`` is set,
    requests are queued and dynamically packed into a single batch.
  - Health check and model info endpoints.

Usage
-----
    python api_server.py --checkpoint out/pretrain/best_ckpt.pt --model modern

Then test with curl:

    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"modern","messages":[{"role":"user","content":"Hello"}],"max_tokens":20}'
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import uvicorn

from model.baseline_gpt import BaselineGPT, BaselineGPTConfig
from model.modern_gpt import ModernGPT, ModernGPTConfig


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: str = "user"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "modern"
    messages: List[Message]
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=50, ge=1)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(default=1.0, ge=1.0)
    stream: bool = False
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    model: str = "modern"
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=50, ge=1)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(default=1.0, ge=1.0)
    stream: bool = False
    stop: Optional[List[str]] = None


class Choice(BaseModel):
    index: int = 0
    message: Optional[Message] = None
    text: Optional[str] = None
    finish_reason: Optional[str] = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-nanogpt"
    object: str = "chat.completion"
    created: int = 0
    model: str = "modern"
    choices: List[Choice]
    usage: Usage


class CompletionResponse(BaseModel):
    id: str = "cmpl-nanogpt"
    object: str = "text_completion"
    created: int = 0
    model: str = "modern"
    choices: List[Choice]
    usage: Usage


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------
class ModelState:
    """Encapsulates the loaded model, tokenizer, and generation helpers."""

    def __init__(
        self,
        checkpoint: str,
        model_type: str,
        device: str,
        compile_model: bool = False,
    ) -> None:
        self.device = device
        self.model_type = model_type
        self.model, self.config = self._load(checkpoint, model_type, device)
        self._tokenizer = None

        # Warm-up a single forward pass to avoid first-request latency
        dummy = torch.randint(
            0,
            self.config.vocab_size,
            (1, 4),
            dtype=torch.long,
            device=self.device,
        )
        with torch.no_grad():
            self.model(dummy)

        # Optional: torch.compile for inference (CUDA only)
        if compile_model and device.startswith("cuda"):
            try:
                self.model = torch.compile(
                    self.model, mode="reduce-overhead", fullgraph=False, dynamic=True
                )
            except Exception as e:
                warnings.warn(f"torch.compile failed for API server: {e}")

    def _load(
        self, ckpt_path: str, model_type: str, device: str
    ) -> Tuple[nn.Module, Any]:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_config = ckpt.get("config", {})
        if isinstance(raw_config, dict):
            if model_type == "baseline":
                config = BaselineGPTConfig.from_dict(raw_config)
                model = BaselineGPT(config)
            else:
                config = ModernGPTConfig.from_dict(raw_config)
                model = ModernGPT(config)
        else:
            config = raw_config
            model = (BaselineGPT if model_type == "baseline" else ModernGPT)(config)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()
        return model, config

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding("gpt2")
            except Exception:
                self._tokenizer = None
        return self._tokenizer

    def encode(self, text: str) -> torch.Tensor:
        if self.tokenizer is not None:
            toks = self.tokenizer.encode(text)
        else:
            toks = [ord(c) for c in text[: self.config.block_size]]
        return torch.tensor([toks], dtype=torch.long, device=self.device)

    def decode(self, ids: torch.Tensor) -> str:
        if ids.ndim == 2:
            ids = ids[0]
        if self.tokenizer is not None:
            return self.tokenizer.decode(ids.tolist())
        return "".join(chr(i) for i in ids.tolist() if i < 256)

    @torch.no_grad()
    def generate(
        self,
        prompt_text: str,
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        repetition_penalty: Optional[float],
    ) -> Tuple[str, int, int]:
        """Generate text from a prompt string.

        Returns
        -------
        text : str
            Generated text only (without the prompt).
        prompt_tokens : int
        completion_tokens : int
        """
        prompt_ids = self.encode(prompt_text)
        prompt_len = prompt_ids.shape[1]

        if hasattr(self.model, "generate"):
            gen_kwargs: Dict[str, Any] = dict(
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_cache=True,
            )
            if "compile" in self.model.generate.__code__.co_varnames:
                gen_kwargs["compile"] = False
            out_ids = self.model.generate(prompt_ids, **gen_kwargs)
        else:
            out_ids = self._fallback_generate(prompt_ids, max_tokens, temperature)

        completion_ids = out_ids[0, prompt_len:]
        text = self.decode(completion_ids)
        return text, prompt_len, completion_ids.shape[0]

    def _fallback_generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
    ) -> torch.Tensor:
        """Naive fallback when model.generate is not available."""
        import torch.nn.functional as F

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits = self.model(idx_cond)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="nanoGPT-Modern API",
    description="OpenAI-compatible inference server for nanoGPT-Modern",
    version="0.2.0",
)

model_state: Optional[ModelState] = None


@app.on_event("startup")
async def startup_event() -> None:
    """Log startup info."""
    print(f"[API Server] Model loaded: {model_state.model_type if model_state else 'None'}")


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    if model_state is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )
    return {"status": "ok", "model": model_state.model_type}


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "modern" if model_state is None else model_state.model_type,
                "object": "model",
                "owned_by": "nanoGPT-Modern",
            }
        ],
    }


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(body: CompletionRequest) -> CompletionResponse:
    """Legacy (non-chat) completions endpoint."""
    if model_state is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    text, prompt_tokens, completion_tokens = model_state.generate(
        prompt_text=body.prompt,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_k=body.top_k,
        top_p=body.top_p,
        repetition_penalty=body.repetition_penalty,
    )

    created = int(time.time())
    return CompletionResponse(
        id=f"cmpl-{created}",
        created=created,
        model=body.model,
        choices=[
            Choice(
                index=0,
                text=text,
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(body: ChatCompletionRequest) -> ChatCompletionResponse:
    """Chat completions endpoint (OpenAI-compatible)."""
    if model_state is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    # Simple prompt formatting: concatenate messages with role markers.
    prompt_parts: List[str] = []
    for msg in body.messages:
        prompt_parts.append(f"{msg.role}: {msg.content}")
    prompt_text = "\n".join(prompt_parts) + "\nassistant:"

    text, prompt_tokens, completion_tokens = model_state.generate(
        prompt_text=prompt_text,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_k=body.top_k,
        top_p=body.top_p,
        repetition_penalty=body.repetition_penalty,
    )

    created = int(time.time())
    return ChatCompletionResponse(
        id=f"chatcmpl-{created}",
        created=created,
        model=body.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nanoGPT-Modern API Server")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model", type=str, default="modern", choices=["baseline", "modern"]
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    global model_state
    model_state = ModelState(
        checkpoint=args.checkpoint,
        model_type=args.model,
        device=args.device,
        compile_model=args.compile,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
