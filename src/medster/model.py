"""
MedsterPi Model - Pure Anthropic SDK Implementation
No LangChain dependencies - direct API calls only.

Based on Pi Agent Framework philosophy:
- Minimal abstraction layers
- Direct SDK usage for clarity and control
- Simple interface: messages in, response out
"""

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import anthropic

# Load environment variables (override empty values)
load_dotenv(override=True)


# ============================================================================
# CLIENT SINGLETON
# ============================================================================

_client: Optional[anthropic.Anthropic] = None


def get_client() -> anthropic.Anthropic:
    """Get or create Anthropic client singleton."""
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ============================================================================
# MAIN LLM CALL FUNCTION
# ============================================================================

def call_llm(
    messages: List[Dict[str, Any]],
    system_prompt: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096
) -> Dict[str, Any]:
    """
    Call Claude LLM with messages and tools.

    This is the core function for the Pi-style event loop.
    Simple interface: messages in, structured response out.

    Args:
        messages: Conversation history in Anthropic format
        system_prompt: System prompt for the model
        tools: Optional list of tool definitions (Anthropic format)
        model: Model identifier
        max_tokens: Maximum tokens in response

    Returns:
        Dict with keys:
            - content: Text content from response (if any)
            - content_blocks: Raw content blocks from response
            - tool_calls: List of tool calls (if any)
            - stop_reason: Why the model stopped
            - usage: Token usage info
    """
    client = get_client()

    # Build request kwargs
    request_kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": messages,
    }

    # Add tools if provided
    if tools:
        request_kwargs["tools"] = tools

    # Make the API call
    response = client.messages.create(**request_kwargs)

    # Parse response into structured format
    result = {
        "content": "",
        "content_blocks": [],
        "tool_calls": [],
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        }
    }

    # Process content blocks
    for block in response.content:
        result["content_blocks"].append(block)

        if block.type == "text":
            result["content"] += block.text

        elif block.type == "tool_use":
            result["tool_calls"].append({
                "id": block.id,
                "name": block.name,
                "input": block.input
            })

    return result


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def simple_completion(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024
) -> str:
    """
    Simple text completion without tools.

    Args:
        prompt: User prompt
        system_prompt: System prompt
        model: Model identifier
        max_tokens: Maximum tokens

    Returns:
        Text response
    """
    messages = [{"role": "user", "content": prompt}]
    response = call_llm(
        messages=messages,
        system_prompt=system_prompt,
        model=model,
        max_tokens=max_tokens
    )
    return response["content"]


def count_tokens(text: str) -> int:
    """
    Estimate token count for text.
    Uses rough approximation: ~4 chars per token.

    Args:
        text: Text to count

    Returns:
        Estimated token count
    """
    return len(text) // 4


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Available models with their characteristics
MODELS = {
    "claude-sonnet-4-20250514": {
        "name": "Claude Sonnet 4",
        "context_window": 200000,
        "max_output": 8192,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "recommended_for": ["general", "coding", "analysis"]
    },
    "claude-opus-4-20250514": {
        "name": "Claude Opus 4",
        "context_window": 200000,
        "max_output": 8192,
        "cost_per_1k_input": 0.015,
        "cost_per_1k_output": 0.075,
        "recommended_for": ["complex reasoning", "research"]
    },
    "claude-3-5-haiku-20241022": {
        "name": "Claude 3.5 Haiku",
        "context_window": 200000,
        "max_output": 8192,
        "cost_per_1k_input": 0.0008,
        "cost_per_1k_output": 0.004,
        "recommended_for": ["fast", "simple tasks"]
    }
}


def get_model_info(model: str) -> Dict[str, Any]:
    """Get information about a model."""
    return MODELS.get(model, MODELS["claude-sonnet-4-20250514"])
