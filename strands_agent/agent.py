"""
strands_agent/agent.py

Strands agent for evaluation with rLLM.
"""

from dotenv import load_dotenv
from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools.browser import LocalChromiumBrowser
import os

# Optional mem0 memory tool from Strands SDK (simple try-import)
try:
    from strands_tools.memory.mem0 import mem0_memory  # type: ignore
except Exception:
    mem0_memory = None  # type: ignore


def build_agent() -> Agent:
    """
    Build Strands agent with official LocalChromiumBrowser tool and OpenAI model.
    """
    load_dotenv()

    browser = LocalChromiumBrowser()
    model = OpenAIModel(model_id="gpt-4o-mini")

    mem0_api_key = os.getenv("MEM0_API_KEY")
    memory_user_id = os.getenv("MEMORY_USER_ID", "default_user")

    tools = [browser.browser]
    if mem0_api_key and mem0_memory is not None:
        tools.append(mem0_memory)
        memory_guidance = (
            f"\nYou have access to a persistent memory tool (mem0). "
            f"When helpful, store user-specific facts and retrieve relevant memories. "
            f"Use user_id='{memory_user_id}'. Respond concisely."
        )
    else:
        memory_guidance = ""

    agent = Agent(
        model=model,
        tools=tools,
        system_prompt=(
            """You are a helpful assistant with web browsing capabilities.
        Use the browser tool to navigate websites and find specific information.
        When asked to find current information like prices or facts, browse relevant websites to get accurate data."""
            + memory_guidance
        ),
    )

    return agent