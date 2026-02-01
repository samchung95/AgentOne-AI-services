"""LLM Service FastAPI Application.

Provides HTTP endpoints for LLM generation:
- POST /v1/generate - Non-streaming generation
- POST /v1/generate-stream - NDJSON streaming generation
- GET /health - Health check
"""

import json
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from services.llm_service.api.protocol import (
    GenerateRequest,
    GenerateResponse,
    ToolCallResponse,
    UsageResponse,
)
from services.llm_service.api.registry import get_registry
from services.llm_service.core.config.settings import get_settings
from services.llm_service.core.llm.base import ImageContent, LLMMessage, LLMToolDefinition, TextContent
from services.llm_service.core.llm.dispatcher import get_dispatcher

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    settings = get_settings()
    # Initialize dispatcher (triggers lazy creation of global singleton)
    dispatcher = get_dispatcher()
    logger.info(
        "llm_service_starting",
        config_profile=settings.config_profile,
        debug=settings.debug,
        dispatcher_config={
            "max_concurrent_requests": dispatcher.config.max_concurrent_requests,
            "queue_timeout_seconds": dispatcher.config.queue_timeout_seconds,
        },
    )
    yield
    # Cleanup: close all cached clients
    registry = get_registry()
    await registry.close_all()
    logger.info("llm_service_shutdown")


app = FastAPI(
    title="AgentOne LLM Service",
    description="Multi-provider LLM gateway with streaming support",
    version="0.1.0",
    lifespan=lifespan,
)


def _convert_messages(messages: list[dict[str, Any]]) -> list[LLMMessage]:
    """Convert request messages to LLMMessage objects."""
    result = []
    for msg in messages:
        content = msg.get("content")

        # Handle multimodal content
        if isinstance(content, list):
            blocks = []
            for block in content:
                if block.get("type") == "text":
                    blocks.append(TextContent(text=block.get("text", "")))
                elif block.get("type") == "image":
                    blocks.append(
                        ImageContent(
                            source_type=block.get("source_type", "base64"),
                            media_type=block.get("media_type", "image/png"),
                            data=block.get("data", ""),
                        )
                    )
            content = blocks

        result.append(
            LLMMessage(
                role=msg["role"],
                content=content,
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id"),
                name=msg.get("name"),
            )
        )
    return result


def _convert_tools(tools: list[dict[str, Any]] | None) -> list[LLMToolDefinition] | None:
    """Convert request tools to LLMToolDefinition objects."""
    if not tools:
        return None

    return [
        LLMToolDefinition(
            name=t["name"],
            description=t["description"],
            parameters=t["parameters"],
            audience=t.get("audience", "internal"),
            scopes=t.get("scopes", []),
        )
        for t in tools
    ]


@app.get("/health")
async def health_check():
    """Health check endpoint.

    Returns healthy status even when rate limited (rate limiting is expected
    behavior, not an unhealthy state).

    Response includes:
    - status: Always "healthy" (service is operational)
    - service: Service name
    - registry: Client registry statistics
    - dispatcher_status: Rate limiting and concurrency status
    - providers: Per-provider health status with circuit breaker state
    """
    registry = get_registry()
    dispatcher = get_dispatcher()
    return {
        "status": "healthy",
        "service": "llm_service",
        "registry": registry.get_stats(),
        "dispatcher_status": dispatcher.get_status(),
        "providers": dispatcher.get_provider_health(),
    }


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate a complete (non-streaming) response.

    Args:
        request: Generation request with messages, tools, and parameters.

    Returns:
        Complete LLM response with content, tool calls, and usage.
    """
    registry = get_registry()
    dispatcher = get_dispatcher()

    try:
        # Resolve model to get provider for dispatcher
        model_info = registry.resolve_model(request.use_case, request.model)
        provider = model_info.provider

        # Acquire dispatcher slot for rate limiting and concurrency control
        async with await dispatcher.acquire(provider):
            client = await registry.get_client(request.use_case, request.model)

            messages = _convert_messages([m.model_dump() for m in request.messages])
            tools = _convert_tools([t.model_dump() for t in request.tools] if request.tools else None)

            response = await client.generate(
                messages=messages,
                tools=tools,
                system_prompt=request.system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

        # Convert to response model
        tool_calls = [
            ToolCallResponse(
                tool_call_id=tc.tool_call_id,
                name=tc.name,
                args=tc.args,
                audience=tc.audience,
                scopes=tc.scopes,
                extensions=tc.extensions,
            )
            for tc in response.tool_calls
        ]

        usage = None
        if response.usage:
            usage = UsageResponse(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens or 0,
                model_name=response.usage.model_name,
            )

        return GenerateResponse(
            content=response.content,
            tool_calls=tool_calls,
            finish_reason=response.finish_reason,
            usage=usage,
        )

    except Exception as e:
        logger.exception("generate_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/generate-stream")
async def generate_stream(request: GenerateRequest):
    """Generate a streaming response (NDJSON).

    Each line is a JSON object with:
    - delta: Text content delta
    - tool_calls: Tool call data (on final chunk)
    - finish_reason: Completion reason (on final chunk)
    - usage: Token usage (on final chunk)
    - error: Error message (on error)

    Args:
        request: Generation request with messages, tools, and parameters.

    Returns:
        StreamingResponse with NDJSON content.
    """
    registry = get_registry()
    dispatcher = get_dispatcher()

    # Resolve model to get provider for dispatcher
    model_info = registry.resolve_model(request.use_case, request.model)
    provider = model_info.provider

    async def stream_generator():
        try:
            # Acquire dispatcher slot for rate limiting and concurrency control
            async with await dispatcher.acquire(provider):
                client = await registry.get_client(request.use_case, request.model)

                messages = _convert_messages([m.model_dump() for m in request.messages])
                tools = _convert_tools([t.model_dump() for t in request.tools] if request.tools else None)

                async for chunk in client.generate_stream(
                    messages=messages,
                    tools=tools,
                    system_prompt=request.system_prompt,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                ):
                    # Build chunk response
                    chunk_data: dict[str, Any] = {}

                    if chunk.delta:
                        chunk_data["delta"] = chunk.delta

                    if chunk.tool_calls:
                        chunk_data["tool_calls"] = chunk.tool_calls

                    if chunk.finish_reason:
                        chunk_data["finish_reason"] = chunk.finish_reason

                    if chunk.usage:
                        chunk_data["usage"] = {
                            "input_tokens": chunk.usage.input_tokens,
                            "output_tokens": chunk.usage.output_tokens,
                            "total_tokens": chunk.usage.total_tokens or 0,
                            "model_name": chunk.usage.model_name,
                        }

                    if chunk_data:
                        yield json.dumps(chunk_data) + "\n"

        except Exception as e:
            logger.exception("stream_error", error=str(e))
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(
        stream_generator(),
        media_type="application/x-ndjson",
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "services.llm_service.main:app",
        host="0.0.0.0",
        port=8001,
        reload=settings.debug,
    )
