from typing import List, Dict, Any

"""
eval/run_gaia.py

Evaluate the Strands-based agent on the GAIA benchmark.  Replace the task loader
with the appropriate code to fetch GAIA tasks and compute metrics.
"""

from rllm.engine.agent_workflow_engine import AgentExecutionEngine
from strands_agent.agent import build_agent


def load_gaia_tasks() -> List[Dict[str, Any]]:
    """
    Load GAIA benchmark tasks.

    GAIA tasks may involve multi-step reasoning and require hidden answers.  Consult
    the GAIA benchmark documentation for details on how to obtain and format
    tasks.  This stub returns a simple list of questions as a placeholder.
    """
    return [
        {"question": "Explain the process of photosynthesis."},
        {"question": "Who was the first president of the United States?"},
    ]


def main() -> None:
    agent = build_agent()
    engine = AgentExecutionEngine(agent=agent)
    tasks = load_gaia_tasks()
    results = engine.execute_tasks(tasks)
    for idx, result in enumerate(results):
        print(f"GAIA Task {idx} result: {result}")


if __name__ == "__main__":
    main()
