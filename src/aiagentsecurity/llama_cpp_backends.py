from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Self

from aicomp_sdk.agents.hf_chat_template.backends.llama_cpp import (
    LlamaCppChatTemplateBackend,
)
from aicomp_sdk.agents.hf_chat_template.types import (
    HFBackendConfig,
    HFGenerationRequest,
)


def _deserialize_function_arguments(raw_arguments: Any) -> Any:
    if not isinstance(raw_arguments, str):
        return raw_arguments
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return raw_arguments
    return parsed if isinstance(parsed, Mapping) else raw_arguments


def _normalize_message_for_gemma_llama_cpp(message: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(message)
    if normalized.get("role") != "assistant" or normalized.get("tool_calls") is None:
        return normalized

    raw_tool_calls = normalized["tool_calls"]
    if not isinstance(raw_tool_calls, list):
        return normalized

    repaired_tool_calls: list[dict[str, Any]] = []
    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, Mapping):
            repaired_tool_calls.append(
                {"type": "function", "function": {"name": "", "arguments": {}}}
            )
            continue
        tool_call = dict(raw_tool_call)
        function_payload = tool_call.get("function")
        if isinstance(function_payload, Mapping):
            repaired_function = dict(function_payload)
            if "arguments" in repaired_function:
                repaired_function["arguments"] = _deserialize_function_arguments(
                    repaired_function["arguments"]
                )
            tool_call["function"] = repaired_function
        repaired_tool_calls.append(tool_call)

    normalized["tool_calls"] = repaired_tool_calls
    return normalized


class GemmaLlamaCppBackend(LlamaCppChatTemplateBackend):
    """llama.cpp backend variant that preserves tool-call argument objects for Gemma."""

    @classmethod
    def from_model_path(
        cls,
        *,
        model_path: str,
        config: HFBackendConfig,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        supports_tools: bool = True,
        llama_cls: Any | None = None,
        llama_kwargs: Mapping[str, Any] | None = None,
    ) -> Self:
        return super().from_model_path(
            model_path=model_path,
            config=config,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            supports_tools=supports_tools,
            llama_cls=llama_cls,
            llama_kwargs=llama_kwargs,
        )

    def _build_completion_kwargs(self, request: HFGenerationRequest) -> dict[str, Any]:
        completion_kwargs = super()._build_completion_kwargs(request)
        raw_messages = completion_kwargs.get("messages", [])
        completion_kwargs["messages"] = [
            _normalize_message_for_gemma_llama_cpp(message)
            if isinstance(message, Mapping)
            else message
            for message in raw_messages
        ]
        return completion_kwargs


# --- gpt-oss (Harmony) multi-hop history fix ---------------------------------
# On the 2nd+ tool hop, the SDK feeds a prior assistant turn back whose ``content``
# still carries raw Harmony control tokens (``<|channel|>analysis<|message|>...``).
# The llama.cpp gpt-oss chat handler rejects channel tags in ``content`` and wants
# analysis in a separate ``thinking`` field. This mirrors the Gemma arguments fix:
# split the channels, keep the final text as content, move analysis to ``thinking``.
_HARMONY_TOKEN_RE = re.compile(r"<\|[^|]*\|>")
_HARMONY_CHANNEL_RE = re.compile(
    r"<\|channel\|>(?P<name>\w+)[^<]*?<\|message\|>(?P<text>.*?)"
    r"(?=<\|(?:channel|end|start|call|return)\|>|$)",
    re.DOTALL,
)


def _normalize_message_for_gpt_oss_llama_cpp(message: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(message)
    if normalized.get("role") != "assistant":
        return normalized
    content = normalized.get("content")
    if not isinstance(content, str) or "<|channel|>" not in content:
        return normalized
    analysis_parts: list[str] = []
    final_parts: list[str] = []
    for match in _HARMONY_CHANNEL_RE.finditer(content):
        text = match.group("text").strip()
        if match.group("name") == "analysis":
            analysis_parts.append(text)
        else:
            final_parts.append(text)
    normalized["content"] = _HARMONY_TOKEN_RE.sub("", " ".join(final_parts)).strip()
    analysis_text = _HARMONY_TOKEN_RE.sub("", " ".join(analysis_parts)).strip()
    if analysis_text:
        normalized["thinking"] = analysis_text
    return normalized


class GptOssLlamaCppBackend(LlamaCppChatTemplateBackend):
    """llama.cpp backend variant that strips Harmony channel tags from assistant
    history so multi-hop gpt-oss traces replay locally."""

    def _build_completion_kwargs(self, request: HFGenerationRequest) -> dict[str, Any]:
        completion_kwargs = super()._build_completion_kwargs(request)
        raw_messages = completion_kwargs.get("messages", [])
        completion_kwargs["messages"] = [
            _normalize_message_for_gpt_oss_llama_cpp(message)
            if isinstance(message, Mapping)
            else message
            for message in raw_messages
        ]
        return completion_kwargs
