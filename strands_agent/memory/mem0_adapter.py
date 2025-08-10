"""
strands_agent/memory/mem0_adapter.py

Adapter to optionally load the Mem0 memory tool from the Strands SDK.

- Tries multiple import locations to maximize compatibility across SDK versions
- Does not require an API key; if `MEM0_API_KEY` is unset, the tool should still load
- Returns `None` if the Mem0 tool is not available in the environment
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Optional

# Environment variable is optional; users can leave it blank
os.environ.setdefault("MEM0_API_KEY", os.getenv("MEM0_API_KEY", ""))


def _try_import(module_name: str, attr_name: Optional[str] = None) -> Optional[Any]:
    try:
        mod = importlib.import_module(module_name)
        if attr_name:
            if hasattr(mod, attr_name):
                return getattr(mod, attr_name)
            return None
        return mod
    except Exception:
        return None


def get_mem0_tool() -> Optional[Any]:
    """
    Attempt to locate and return the Mem0 tool from the Strands SDK.

    Returns:
        The Mem0 tool object (callable or AgentTool) if found, otherwise None.
    """
    candidates: list[tuple[str, Optional[str]]] = [
        # Common expected locations/symbols
        ("strands_tools.mem0_memory", "mem0_memory"),
        ("strands_tools.mem0", "mem0_memory"),
        ("strands_tools.memory.mem0_memory", "mem0_memory"),
        ("strands_tools.memory.mem0", "mem0_memory"),
        # Some packages expose the tool directly as the module
        ("strands_tools.mem0_memory", None),
        ("strands_tools.mem0", None),
    ]

    for module_name, attr_name in candidates:
        tool = _try_import(module_name, attr_name)
        if tool is not None:
            return tool

    return None