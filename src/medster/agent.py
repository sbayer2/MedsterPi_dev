"""
MedsterPi Agent - Pi-Style Event Loop Architecture
Based on Pi Agent Framework by Mario Zechner (@mariozechner)

Key characteristics:
- Simple event-driven loop (not a state machine)
- Messages list is the only state
- Model decides everything autonomously
- Full observability via event callbacks
- Cross-provider handoff support (future)

Architecture:
    User Query
        ↓
    ┌─────────────────────────────────┐
    │  Event Loop                     │
    │  ┌───────────────────────────┐  │
    │  │ 1. Call LLM with messages │  │
    │  │ 2. If tool_use → execute  │  │
    │  │ 3. Add result to messages │  │
    │  │ 4. Repeat until end_turn  │  │
    │  └───────────────────────────┘  │
    └─────────────────────────────────┘
        ↓
    Final Response
"""

import json
import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from medster.model import call_llm
from medster.prompts import get_system_prompt, get_tools_schema
from medster.tools import execute_tool


# ============================================================================
# EVENT TYPES - For observability
# ============================================================================

class EventType(Enum):
    """Event types for the agent loop - enables full observability."""
    LOOP_START = "loop_start"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    LOOP_END = "loop_end"
    ERROR = "error"


@dataclass
class Event:
    """Event emitted during agent execution."""
    type: EventType
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    iteration: int = 0


# Type alias for event handler
EventHandler = Callable[[Event], None]


# ============================================================================
# AGENT CLASS - Pi-Style Event Loop
# ============================================================================

class Agent:
    """
    MedsterPi Agent - Minimal event-driven clinical analysis agent.

    Unlike LangGraph-style agents with complex state machines, this agent
    uses a simple loop:
        while has_tool_calls:
            response = llm(messages)
            results = execute_tools(response.tool_calls)
            messages.append(results)

    The model decides everything - when to call tools, which tools, when to stop.
    """

    def __init__(
        self,
        max_iterations: int = 25,
        model: str = "claude-sonnet-4-5-20250929",
        event_handler: Optional[EventHandler] = None,
        verbose: bool = False
    ):
        """
        Initialize the Pi-style agent.

        Args:
            max_iterations: Safety limit on loop iterations
            model: Anthropic model to use
            event_handler: Optional callback for observability
            verbose: Print debug info to console
        """
        self.max_iterations = max_iterations
        self.model = model
        self.event_handler = event_handler
        self.verbose = verbose
        self.system_prompt = get_system_prompt()
        self.tools = get_tools_schema()

    def _emit(self, event_type: EventType, data: Dict[str, Any], iteration: int = 0):
        """Emit an event for observability."""
        event = Event(type=event_type, data=data, iteration=iteration)

        if self.verbose:
            self._print_event(event)

        if self.event_handler:
            self.event_handler(event)

    def _print_event(self, event: Event):
        """Print event to console for debugging."""
        prefix = f"[{event.iteration}]" if event.iteration > 0 else "[*]"

        if event.type == EventType.LOOP_START:
            print(f"\n{prefix} === MEDSTERPI AGENT START ===")
            print(f"    Query: {event.data.get('query', '')[:100]}...")

        elif event.type == EventType.LLM_REQUEST:
            print(f"{prefix} → LLM Request ({event.data.get('message_count', 0)} messages)")

        elif event.type == EventType.LLM_RESPONSE:
            stop = event.data.get('stop_reason', 'unknown')
            tools = event.data.get('tool_calls', [])
            print(f"{prefix} ← LLM Response (stop={stop}, tools={len(tools)})")

        elif event.type == EventType.TOOL_CALL:
            name = event.data.get('tool_name', 'unknown')
            print(f"{prefix}   🔧 Tool: {name}")

        elif event.type == EventType.TOOL_RESULT:
            name = event.data.get('tool_name', 'unknown')
            success = event.data.get('success', False)
            status = "✓" if success else "✗"
            print(f"{prefix}   {status} Result: {name}")

        elif event.type == EventType.LOOP_END:
            iterations = event.data.get('iterations', 0)
            print(f"{prefix} === AGENT COMPLETE ({iterations} iterations) ===\n")

        elif event.type == EventType.ERROR:
            error = event.data.get('error', 'Unknown error')
            print(f"{prefix} ❌ ERROR: {error}")

    def run(self, query: str) -> str:
        """
        Execute the agent loop.

        This is the core Pi-style event loop:
        1. Send messages to LLM
        2. If LLM wants to use tools, execute them
        3. Add results to messages
        4. Repeat until LLM stops requesting tools

        Args:
            query: The user's clinical query

        Returns:
            The final text response from the agent
        """
        # Initialize messages with user query
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": query}
        ]

        self._emit(EventType.LOOP_START, {"query": query})

        iteration = 0
        final_response = ""

        while iteration < self.max_iterations:
            iteration += 1

            # ─────────────────────────────────────────────────────────────
            # STEP 1: Call LLM
            # ─────────────────────────────────────────────────────────────
            self._emit(EventType.LLM_REQUEST, {
                "message_count": len(messages),
                "iteration": iteration
            }, iteration)

            try:
                response = call_llm(
                    messages=messages,
                    system_prompt=self.system_prompt,
                    tools=self.tools,
                    model=self.model
                )
            except Exception as e:
                self._emit(EventType.ERROR, {"error": str(e)}, iteration)
                return f"Error calling LLM: {str(e)}"

            self._emit(EventType.LLM_RESPONSE, {
                "stop_reason": response.get("stop_reason"),
                "tool_calls": response.get("tool_calls", []),
                "has_content": bool(response.get("content"))
            }, iteration)

            # ─────────────────────────────────────────────────────────────
            # STEP 2: Check if we're done (no tool calls)
            # ─────────────────────────────────────────────────────────────
            tool_calls = response.get("tool_calls", [])

            if not tool_calls:
                # Model is done - extract final response
                final_response = response.get("content", "")
                break

            # ─────────────────────────────────────────────────────────────
            # STEP 3: Execute tools and collect results
            # ─────────────────────────────────────────────────────────────
            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": response.get("content_blocks", [])
            })

            # Execute each tool and collect results
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_id = tool_call.get("id")
                tool_input = tool_call.get("input", {})

                self._emit(EventType.TOOL_CALL, {
                    "tool_name": tool_name,
                    "tool_id": tool_id,
                    "tool_input": tool_input
                }, iteration)

                # Execute the tool
                try:
                    result = execute_tool(tool_name, tool_input)
                    success = True
                    result_content = json.dumps(result) if not isinstance(result, str) else result
                except Exception as e:
                    success = False
                    result_content = json.dumps({"error": str(e)})

                self._emit(EventType.TOOL_RESULT, {
                    "tool_name": tool_name,
                    "tool_id": tool_id,
                    "success": success,
                    "result_preview": result_content[:200] if len(result_content) > 200 else result_content
                }, iteration)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_content
                })

            # ─────────────────────────────────────────────────────────────
            # STEP 4: Add tool results to messages and continue loop
            # ─────────────────────────────────────────────────────────────
            messages.append({
                "role": "user",
                "content": tool_results
            })

        # Check if we hit max iterations
        if iteration >= self.max_iterations:
            self._emit(EventType.ERROR, {
                "error": f"Max iterations ({self.max_iterations}) reached"
            }, iteration)
            final_response = response.get("content", "") + \
                f"\n\n[Note: Analysis stopped after {self.max_iterations} iterations]"

        self._emit(EventType.LOOP_END, {
            "iterations": iteration,
            "response_length": len(final_response)
        }, iteration)

        return final_response


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_agent(
    verbose: bool = True,
    max_iterations: int = 25,
    model: str = "claude-sonnet-4-5-20250929"
) -> Agent:
    """
    Create a MedsterPi agent with sensible defaults.

    Args:
        verbose: Print debug output
        max_iterations: Max loop iterations
        model: Anthropic model to use

    Returns:
        Configured Agent instance
    """
    return Agent(
        max_iterations=max_iterations,
        model=model,
        verbose=verbose
    )


def run_query(query: str, verbose: bool = True) -> str:
    """
    Quick helper to run a single query.

    Args:
        query: Clinical query to analyze
        verbose: Print debug output

    Returns:
        Agent's response
    """
    agent = create_agent(verbose=verbose)
    return agent.run(query)


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    # Load environment variables from .env
    load_dotenv()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What labs are available for patient 12345?"

    print(f"Query: {query}\n")
    response = run_query(query, verbose=True)
    print(f"\nResponse:\n{response}")
