"""
strands_agent/agent.py

This module defines a simple Strands-based agent that uses the default mem0 memory
implementation.  It is provided as a starting point for integrating with
rLLM's AgentWorkflowEngine.  You need to have the `strands` library installed
in your Python environment.

Example usage:

    from strands_agent.agent import build_agent
    agent = build_agent()

Then pass ``agent`` to an rLLM workflow engine for inference and evaluation.
"""

from typing import Any, Dict, List

def build_agent() -> Any:
    """
    Build and return a Strands agent instance configured with a baseline memory
    and no additional tools.  Replace the memory and tools as needed.

    Returns:
        An instance of a Strands agent.
    """
    try:
        # Import Strands classes.  These imports assume you installed the SDK from
        # https://github.com/aws/strands.  Adjust the import paths if the API differs.
        from strands.core import AgentConfig, Agent
        # The default mem0 memory class lives in strands.tools.memory.mem0_memory
        from strands.tools.memory.mem0_memory import Mem0Memory

        # If you need additional tools (e.g., Python or WebSearch), import them here.
        # from strands.tools.python_tool import PythonTool
        # from strands.tools.websearch_tool import WebSearchTool

    except ImportError as exc:
        raise ImportError(
            "Strands SDK is not installed.  "
            "Please clone and install it from https://github.com/aws/strands"
        ) from exc

    # Configure the model provider.  For example, you can use OpenAI GPT‑4o mini via API,
    # or a locally hosted model.  Replace or extend this dict based on your provider.
    model_provider: Dict[str, Any] = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        # The API key can be set via the environment or provided here.
        # "api_key": "sk-...",
    }

    # Instantiate the baseline memory tool (mem0).  To use your own memory,
    # replace this with an instance of your custom memory class (e.g. Mem1Memory).
    memory_tool = Mem0Memory()

    # Define optional additional tools for the agent.  The list can include
    # things like a Python execution environment or web search tool.
    tools: List[Any] = [
        # PythonTool(),
        # WebSearchTool(),
    ]

    # Create the agent configuration.  The AgentConfig API may evolve;
    # refer to the Strands documentation for more fields and defaults.
    config = AgentConfig(
        model=model_provider,
        memory=memory_tool,
        tools=tools,
        enable_react=True,  # Enable ReAct (thought-action loops) if supported.
    )

    # Instantiate and return the agent.  This agent can later be used within
    # an rLLM workflow engine for inference and evaluation.
    agent = Agent(config)
    return agent