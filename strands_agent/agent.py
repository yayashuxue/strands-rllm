"""
strands_agent/agent.py

Strands agent for evaluation with rLLM.
"""

from dotenv import load_dotenv
from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools.browser import LocalChromiumBrowser
import os

def build_agent() -> Agent:
    """
    Build Strands agent with official LocalChromiumBrowser tool and OpenAI model.
    
    Returns:
        A Strands Agent with browser capabilities.
    """
    load_dotenv()
    
    # Create official LocalChromiumBrowser instance
    browser = LocalChromiumBrowser()
    
    # Use OpenAI model instead of default Bedrock
    model = OpenAIModel(model_id="gpt-4o-mini")
    
    agent = Agent(
        model=model,
        tools=[browser.browser],
        system_prompt="""You are a helpful assistant with web browsing capabilities.
        Use the browser tool to navigate websites and find specific information.
        When asked to find current information like prices or facts, browse relevant websites to get accurate data."""
    )
    
    return agent