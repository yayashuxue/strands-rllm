"""
strands_agent/agent.py

This module defines a Strands-based agent using the official Strands SDK.
"""

import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from strands import Agent

def build_agent() -> Agent:
    """
    Build and return a Strands agent instance.
    
    Returns:
        A Strands Agent instance.
    """
    # Load environment variables
    load_dotenv()
    
    # Create a Strands agent
    agent = Agent()
    
    return agent