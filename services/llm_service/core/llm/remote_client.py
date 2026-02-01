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
        self._external_client = client
        self._client_instance: httpx.AsyncClient | None = None
        self._owns_client = client is None

    def _get_endpoint(self) -> str:
        """Return the configured remote service base URL.

        Returns:
            The base URL for the remote LLM service.
        """
        return self._base_url

    def _create_client_instance(
        self, credentials: dict[str, Any], endpoint: str
    ) -> httpx.AsyncClient:
        """Create the httpx.AsyncClient instance.

        Args:
            credentials: Not used for remote client (authentication handled by remote).
            endpoint: Not used directly here (stored in _base_url).

        Returns:
            Configured httpx.AsyncClient.
        """
        return httpx.AsyncClient(timeout=self._timeout_seconds)

    def _initialize_client(self) -> httpx.AsyncClient:
        """Initialize the HTTP client.

        If an external client was provided at construction, use that.
        Otherwise, create a new httpx.AsyncClient with configured timeout.

        Returns:
            The httpx.AsyncClient instance.
        """
        if self._external_client is not None:
            self._owns_client = False
            return self._external_client

        self._owns_client = True
        return self._create_client_instance({}, self._base_url)

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
        http_client = self.client

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
        async with http_client.stream("POST", url, json=payload) as response:
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
        http_client = self.client

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
        response = await http_client.post(url, json=payload)
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
        if self._client_instance and self._owns_client:
            await self._client_instance.aclose()
        self._client_instance = None
