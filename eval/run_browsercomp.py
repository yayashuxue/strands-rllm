from typing import List, Dict, Any

"""
eval/run_browsercomp.py

Evaluate the Strands-based agent on the BrowserComp benchmark using rLLM's
AgentExecutionEngine.  This script is a template; replace the dummy task list
with actual BrowserComp tasks and compute appropriate metrics.
"""

from rllm.engine.agent_workflow_engine import AgentExecutionEngine
from strands_agent.agent import build_agent


def load_tasks() -> List[Dict[str, Any]]:
    """
    Load the BrowserComp benchmark tasks.

    Replace this stub with code to read the dataset (e.g. from a JSON file or API).
    """
    return [
        {"question": "Find the URL of the official website for the Eiffel Tower."},
        {"question": "Look up the current CEO of Amazon and provide their name."},
    ]


def main() -> None:
    agent = build_agent()
    engine = AgentExecutionEngine(agent=agent)
    tasks = load_tasks()
    results = engine.execute_tasks(tasks)
    for idx, result in enumerate(results):
        print(f"BrowserComp Task {idx} result: {result}")


if __name__ == "__main__":
    main()
