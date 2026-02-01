#!/usr/bin/env python3
"""Generate environment variable documentation from Settings classes.

This script introspects the Settings class and all nested settings classes
to generate a markdown documentation file with all configuration options.

Usage:
    python scripts/generate_env_docs.py

Output:
    docs/configuration.md
"""
# ruff: noqa: E402 - imports after sys.path modification are intentional

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args, get_origin

# Add the repository root to the path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings

from services.llm_service.core.config.settings import (
    AzureOpenAISettings,
    GenAIPlatformSettings,
    OpenAISettings,
    OpenRouterSettings,
    Settings,
    TelemetrySettings,
    VertexAISettings,
)


def get_env_var_name(
    field_name: str,
    field_info: FieldInfo,
    model_config: dict[str, Any],
) -> str:
    """Determine the environment variable name for a field.

    Priority:
    1. validation_alias if set
    2. env_prefix + field_name (uppercased)
    """
    # Check for explicit validation_alias
    if field_info.validation_alias:
        return str(field_info.validation_alias)

    # Use env_prefix from model_config
    env_prefix = model_config.get("env_prefix", "")
    return f"{env_prefix}{field_name}".upper()


def format_type(annotation: Any) -> str:
    """Format a type annotation as a readable string."""
    if annotation is None:
        return "None"

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Handle Union types (e.g., str | None)
    if origin is type(str | None):
        # Filter out NoneType
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return f"{format_type(non_none_args[0])} | None"
        return " | ".join(format_type(arg) for arg in non_none_args) + " | None"

    # Handle Literal types
    if hasattr(annotation, "__origin__") and str(annotation.__origin__) == "typing.Literal":
        values = ", ".join(repr(v) for v in get_args(annotation))
        return f"Literal[{values}]"

    # Handle basic types
    if hasattr(annotation, "__name__"):
        return annotation.__name__

    return str(annotation)


def format_default(default: Any) -> str:
    """Format a default value for display."""
    if default is None:
        return "(none)"
    if isinstance(default, bool):
        return str(default).lower()
    if isinstance(default, str):
        if not default:
            return '""'
        # Truncate very long defaults
        if len(default) > 50:
            return f'"{default[:47]}..."'
        return f'"{default}"'
    return str(default)


def extract_fields(
    settings_class: type[BaseSettings],
) -> list[dict[str, str]]:
    """Extract field information from a settings class."""
    fields = []
    model_config = getattr(settings_class, "model_config", {})

    for field_name, field_info in settings_class.model_fields.items():
        # Skip nested settings classes (handled separately)
        annotation = field_info.annotation
        if annotation and isinstance(annotation, type) and issubclass(annotation, BaseModel):
            continue

        env_var = get_env_var_name(field_name, field_info, model_config)
        type_str = format_type(annotation)
        default_str = format_default(field_info.default)
        description = field_info.description or ""

        fields.append(
            {
                "env_var": env_var,
                "type": type_str,
                "default": default_str,
                "description": description,
            }
        )

    return fields


def generate_table(fields: list[dict[str, str]]) -> str:
    """Generate a markdown table from field information."""
    if not fields:
        return "_No configuration options._\n"

    lines = [
        "| Variable | Type | Default | Description |",
        "|----------|------|---------|-------------|",
    ]

    for field in fields:
        # Escape pipe characters in values
        env_var = field["env_var"].replace("|", "\\|")
        type_str = field["type"].replace("|", "\\|")
        default = field["default"].replace("|", "\\|")
        description = field["description"].replace("|", "\\|")
        lines.append(f"| `{env_var}` | `{type_str}` | {default} | {description} |")

    return "\n".join(lines) + "\n"


def generate_documentation() -> str:
    """Generate the full configuration documentation."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    sections = []

    # Header
    sections.append(
        f"""# Configuration Reference

This document lists all environment variables used to configure the LLM Service.

> **Generated automatically** by `scripts/generate_env_docs.py` on {timestamp}.
> Do not edit this file directly.

## Table of Contents

- [Application Settings](#application-settings)
- [OpenAI Settings](#openai-settings)
- [OpenRouter Settings](#openrouter-settings)
- [Azure OpenAI Settings](#azure-openai-settings)
- [Vertex AI Settings](#vertex-ai-settings)
- [GenAI Platform Settings](#genai-platform-settings)
- [Telemetry Settings](#telemetry-settings)

---
"""
    )

    # Application Settings (top-level Settings class, excluding nested)
    sections.append("## Application Settings\n")
    sections.append("Core application configuration.\n\n")
    app_fields = extract_fields(Settings)
    sections.append(generate_table(app_fields))

    # Provider sections
    provider_configs = [
        ("OpenAI Settings", OpenAISettings, "Configuration for direct OpenAI API access."),
        ("OpenRouter Settings", OpenRouterSettings, "Configuration for OpenRouter API access."),
        (
            "Azure OpenAI Settings",
            AzureOpenAISettings,
            "Configuration for Azure OpenAI Service.",
        ),
        (
            "Vertex AI Settings",
            VertexAISettings,
            "Configuration for Google Vertex AI / Gemini.",
        ),
        (
            "GenAI Platform Settings",
            GenAIPlatformSettings,
            "Configuration for the company GenAI Platform gateway.",
        ),
        (
            "Telemetry Settings",
            TelemetrySettings,
            "Configuration for observability and telemetry.",
        ),
    ]

    for title, settings_class, description in provider_configs:
        sections.append(f"\n## {title}\n")
        sections.append(f"{description}\n\n")
        fields = extract_fields(settings_class)
        sections.append(generate_table(fields))

    # Footer
    sections.append(
        """
---

## Usage Notes

1. **Environment Variables**: All settings can be configured via environment variables.
   Set them in your shell, `.env` file, or container configuration.

2. **Nested Settings**: Provider-specific settings are grouped by prefix:
   - `OPENAI_*` for OpenAI
   - `OPENROUTER_*` for OpenRouter
   - `AZURE_OPENAI_*` for Azure OpenAI
   - `VERTEX_*` for Vertex AI
   - `GENAI_PLATFORM_*` for GenAI Platform

3. **Configuration Profile**: The `CONFIG_PROFILE` variable determines which
   `config/models.{profile}.yaml` file is loaded for model configuration.

4. **Optional vs Required**: Fields with `(none)` as default are optional.
   Required fields must be set for the corresponding provider to work.
"""
    )

    return "\n".join(sections)


def main() -> None:
    """Generate environment documentation and write to docs/configuration.md."""
    output_path = REPO_ROOT / "docs" / "configuration.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = generate_documentation()
    output_path.write_text(content, encoding="utf-8")

    print(f"Documentation generated: {output_path}")
    print(f"Total size: {len(content)} bytes")


if __name__ == "__main__":
    main()
