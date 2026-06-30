"""Chat templates for multi-turn dialogue formatting.

Supports multiple popular formats:
  * ``chatml`` — OpenAI ChatML (`<|im_start|>user`, `<|im_end|>`, etc.)
  * ``llama-2`` — Meta LLaMA-2 style (`[INST] <<SYS>> ... <</SYS>> ... [/INST]`)
  * ``gemma`` — Google Gemma style (`<start_of_turn>user\n...<end_of_turn>`)

Each template implements:
  * ``apply_chat_template(messages) -> str``
  * ``build_kv_cache_prefix(model, device) -> past_kvs`` (optional) for system-prompt
    KV cache pre-computation and reuse.

Usage
-----
>>> from data.chat_templates import ChatTemplate
>>> tmpl = ChatTemplate.from_name("chatml")
>>> text = tmpl.apply_chat_template([
...     {"role": "system", "content": "You are a helpful assistant."},
...     {"role": "user", "content": "Hello!"},
... ])
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclasses.dataclass
class Message:
    """Single chat message."""

    role: str
    content: str


class ChatTemplate(ABC):
    """Abstract base class for chat templates."""

    @abstractmethod
    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Convert a list of messages into a single formatted string."""
        raise NotImplementedError

    @abstractmethod
    def get_stop_strings(self) -> List[str]:
        """Return the list of strings that terminate the assistant turn."""
        raise NotImplementedError

    def build_kv_cache_prefix(
        self,
        model: nn.Module,
        system_message: Optional[Dict[str, str]] = None,
        device: str = "cuda",
    ) -> Optional[Any]:
        """Pre-compute the KV cache for a fixed system prompt.

        If the model supports ``use_cache=True``, this runs a forward pass over
        the system prompt and returns the ``past_kvs`` tuple so that subsequent
        user-assistant turns can skip re-computing the system prompt encoding.

        Returns
        -------
        past_kvs or None
        """
        if system_message is None:
            return None
        text = self.apply_chat_template([system_message])
        try:
            import tiktoken

            tokenizer = tiktoken.get_encoding("gpt2")
            ids = tokenizer.encode(text)
        except Exception:
            ids = [ord(c) for c in text]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        model.eval()
        with torch.no_grad():
            _, _, past_kvs = model(idx, use_cache=True, start_pos=0)
        return past_kvs

    @classmethod
    def from_name(cls, name: str) -> "ChatTemplate":
        """Factory: return the template instance for ``name``."""
        registry = {
            "chatml": ChatMLTemplate,
            "llama-2": Llama2Template,
            "gemma": GemmaTemplate,
        }
        if name.lower() not in registry:
            raise ValueError(
                f"Unknown chat template: {name}. Available: {list(registry.keys())}"
            )
        return registry[name.lower()]()  # type: ignore[abstract]


class ChatMLTemplate(ChatTemplate):
    """OpenAI ChatML format.

    Format::

        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Hello!<|im_end|>
        <|im_start|>assistant

    The assistant turn is open-ended; generation stops at the next
    ``<|im_end|>`` token.
    """

    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{self.IM_START}{role}\n{content}{self.IM_END}")
        # If the last message is not from assistant, append the assistant prefix.
        if not messages or messages[-1].get("role") != "assistant":
            parts.append(f"{self.IM_START}assistant\n")
        return "\n".join(parts)

    def get_stop_strings(self) -> List[str]:
        return [self.IM_END]

    def build_kv_cache_prefix(
        self,
        model: nn.Module,
        system_message: Optional[Dict[str, str]] = None,
        device: str = "cuda",
    ) -> Optional[Any]:
        if system_message is None:
            return None
        text = self.apply_chat_template([system_message])
        try:
            import tiktoken

            tokenizer = tiktoken.get_encoding("gpt2")
            ids = tokenizer.encode(text)
        except Exception:
            ids = [ord(c) for c in text]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        model.eval()
        with torch.no_grad():
            _, _, past_kvs = model(idx, use_cache=True, start_pos=0)
        return past_kvs


class Llama2Template(ChatTemplate):
    """Meta LLaMA-2 chat format.

    Format::

        [INST] <<SYS>>
        You are a helpful assistant.
        <</SYS>>
        Hello! [/INST]

    System prompt is injected inside the first user INST block.
    """

    SYS_START = "<<SYS>>"
    SYS_END = "<</SYS>>"
    INST_START = "[INST]"
    INST_END = "[/INST]"

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        parts = []
        system_text = ""
        user_assistant_pairs = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "user")
            if role == "system":
                system_text = msg.get("content", "")
                i += 1
            elif role == "user":
                user_content = msg.get("content", "")
                i += 1
                assistant_content = ""
                if i < len(messages) and messages[i].get("role") == "assistant":
                    assistant_content = messages[i].get("content", "")
                    i += 1
                user_assistant_pairs.append((user_content, assistant_content))
            else:
                i += 1

        for user, assistant in user_assistant_pairs:
            if system_text:
                inst = f"{self.INST_START} {self.SYS_START}\n{system_text}\n{self.SYS_END}\n\n{user} {self.INST_END}"
                system_text = ""  # only on first turn
            else:
                inst = f"{self.INST_START} {user} {self.INST_END}"
            parts.append(inst)
            if assistant:
                parts.append(f" {assistant} ")

        # If the last turn has no assistant response, leave it open.
        if user_assistant_pairs and not user_assistant_pairs[-1][1]:
            pass  # already open
        return "".join(parts)

    def get_stop_strings(self) -> List[str]:
        return [self.INST_END]


class GemmaTemplate(ChatTemplate):
    """Google Gemma chat format.

    Format::

        <start_of_turn>user
        Hello!<end_of_turn>
        <start_of_turn>model

    """

    START_TURN = "<start_of_turn>"
    END_TURN = "<end_of_turn>"

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Gemma uses "model" for assistant.
            if role == "assistant":
                role = "model"
            parts.append(f"{self.START_TURN}{role}\n{content}\n{self.END_TURN}")
        if not messages or messages[-1].get("role") != "assistant":
            parts.append(f"{self.START_TURN}model\n")
        return "\n".join(parts)

    def get_stop_strings(self) -> List[str]:
        return [self.END_TURN]


def apply_chat_template(
    messages: List[Dict[str, str]], template: str = "chatml"
) -> str:
    """Convenience function: format messages with a named template.

    Parameters
    ----------
    messages : list[dict]
        Each dict has ``role`` and ``content`` keys.
    template : str
        One of ``chatml``, ``llama-2``, ``gemma``.

    Returns
    -------
    formatted : str
    """
    tmpl = ChatTemplate.from_name(template)
    return tmpl.apply_chat_template(messages)


def inject_system_prompt_kv_cache(
    model: nn.Module,
    system_prompt: str,
    template: str = "chatml",
    device: str = "cuda",
) -> Optional[Any]:
    """Pre-compute KV cache for a system prompt so it can be reused across turns.

    Returns
    -------
    past_kvs or None
        The KV cache tuple to pass as ``past_kvs`` in subsequent generation calls.
    """
    tmpl = ChatTemplate.from_name(template)
    return tmpl.build_kv_cache_prefix(
        model,
        system_message={"role": "system", "content": system_prompt},
        device=device,
    )
