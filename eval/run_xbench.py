from typing import List, Dict, Any

"""
eval/run_xbench.py

Evaluate the Strands-based agent on the XBench benchmark.  XBench is designed to
test agents on multi-step tasks in professional settings (e.g., recruiting or
marketing workflows).  Replace the task loader with appropriate code to fetch
XBench tasks and compute the relevant metrics.
"""

from rllm.engine.agent_workflow_engine import AgentExecutionEngine
from strands_agent.agent import build_agent


def load_xbench_tasks() -> List[Dict[str, Any]]:
    """
    Load XBench benchmark tasks.

    XBench tasks are scenario-based and may require the agent to perform
    sequences of actions with external tools.  This stub returns simple
    placeholder questions.
    """
    return [
        {"question": "Draft a job description for a software engineer position."},
        {"question": "Identify three marketing channels suitable for a B2B SaaS product."},
    ]


def main() -> None:
    agent = build_agent()
    engine = AgentExecutionEngine(agent=agent)
    tasks = load_xbench_tasks()
    results = engine.execute_tasks(tasks)
    for idx, result in enumerate(results):
        print(f"XBench Task {idx} result: {result}")


if __name__ == "__main__":
    main()
