"""Remote LLM client that delegates generation to the LLM service over HTTP."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from services.llm_service.core.llm.base import BaseLLMClient, LLMChunk, LLMMessage, LLMResponse, LLMToolDefinition
from shared.protocol.common import Usage
from shared.protocol.tool_models import ToolCall


class RemoteLLMClient(BaseLLMClient):
    """BaseLLMClient implementation backed by a remote LLM service."""

    def __init__(
        self,
        base_url: str,
        use_case: str = "chat",
        model: str | None = None,
        timeout_seconds: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._use_case = use_case
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._client = client
        self._owns_client = client is None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_seconds)
            self._owns_client = True
        return self._client

    @staticmethod
    def _serialize_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for msg in messages:
            data: dict[str, Any] = {"role": msg.role}

            if msg.content is None or isinstance(msg.content, str):
                data["content"] = msg.content
            else:
                blocks = []
                for block in msg.content:
                    if hasattr(block, "model_dump"):
                        blocks.append(block.model_dump())
                    else:
                        blocks.append(block)
                data["content"] = blocks

            if msg.tool_calls is not None:
                data["tool_calls"] = msg.tool_calls
            if msg.tool_call_id is not None:
                data["tool_call_id"] = msg.tool_call_id
            if msg.name is not None:
                data["name"] = msg.name

            serialized.append(data)
        return serialized

    @staticmethod
    def _serialize_tools(tools: list[LLMToolDefinition] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "audience": t.audience,
                "scopes": t.scopes,
            }
            for t in tools
        ]

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        client = self._ensure_client()

        payload: dict[str, Any] = {
            "use_case": self._use_case,
            "model": self._model,
            "messages": self._serialize_messages(messages),
            "tools": self._serialize_tools(tools),
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        url = f"{self._base_url}/v1/generate-stream"
        async with client.stream("POST", url, json=payload) as response:
            if response.is_error:
                body = await response.aread()
                detail = body.decode("utf-8", errors="replace")
                raise httpx.HTTPStatusError(
                    f"LLM service error {response.status_code} for url '{url}': {detail[:500]}",
                    request=response.request,
                    response=response,
                )

            async for line in response.aiter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and obj.get("error"):
                    raise RuntimeError(obj.get("error"))

                usage = Usage.model_validate(obj["usage"]) if isinstance(obj, dict) and obj.get("usage") else None
                yield LLMChunk(
                    delta=obj.get("delta", "") if isinstance(obj, dict) else "",
                    tool_calls=obj.get("tool_calls") if isinstance(obj, dict) else None,
                    finish_reason=obj.get("finish_reason") if isinstance(obj, dict) else None,
                    usage=usage,
                )

    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        client = self._ensure_client()

        payload: dict[str, Any] = {
            "use_case": self._use_case,
            "model": self._model,
            "messages": self._serialize_messages(messages),
            "tools": self._serialize_tools(tools),
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        url = f"{self._base_url}/v1/generate"
        response = await client.post(url, json=payload)
        if response.is_error:
            detail = response.text
            raise httpx.HTTPStatusError(
                f"LLM service error {response.status_code} for url '{url}': {detail[:500]}",
                request=response.request,
                response=response,
            )

        data = response.json() if response.content else {}
        usage = Usage.model_validate(data["usage"]) if data.get("usage") else None
        tool_calls = [ToolCall.model_validate(tc) for tc in data.get("tool_calls", [])]

        return LLMResponse(
            content=data.get("content", "") or "",
            tool_calls=tool_calls,
            finish_reason=data.get("finish_reason", "stop") or "stop",
            usage=usage,
        )

    async def close(self) -> None:
        if self._client and self._owns_client:
            await self._client.aclose()
        self._client = None
