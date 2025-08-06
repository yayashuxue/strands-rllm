from typing import List, Dict, Any

"""
rllm_workflow/workflow.py

Demonstrate how to use rLLM's AgentWorkflowEngine to run a Strands agent over a set of tasks.

To run this script:
    python rllm_workflow/workflow.py

Replace the dummy task list with your own dataset or benchmark loader.
"""

try:
    from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
except ImportError as exc:
    raise ImportError(
        "rLLM is not installed. Please clone and install it from https://github.com/agentica-project/rllm"
    ) from exc

from strands_agent.agent import build_agent


def main() -> None:
    """
    Build the Strands agent and run a simple evaluation loop using rLLM.
    """
    agent_instance = build_agent()
    engine = AgentWorkflowEngine(agent=agent_instance)

    tasks: List[Dict[str, Any]] = [
        {"question": "What is the capital of France?"},
        {"question": "Who wrote the novel '1984'?"},
    ]
    results = engine.execute_tasks(tasks)

    for idx, result in enumerate(results):
        print(f"Task {idx} result: {result}")


if __name__ == "__main__":
    main()
