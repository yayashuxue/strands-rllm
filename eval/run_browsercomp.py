from __future__ import annotations
from typing import List, Dict, Any
import os
import asyncio
from pathlib import Path

from dotenv import load_dotenv

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore

from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine
from rllm_workflow.strands_agent_wrapper import StrandsAgentWrapper
from rllm_workflow.strands_env import StrandsEnv


"""
Evaluate the Strands-based agent on a small BrowserComp-like toy set using rLLM's
AsyncAgentExecutionEngine. This wires the same config as rllm_workflow/workflow.py
so it runs out-of-the-box with an OpenAI API key.

Replace `load_tasks()` with real BrowserComp tasks and compute proper metrics.
"""


def load_tasks() -> List[Dict[str, Any]]:
    # Toy tasks shaped like BrowserComp "question" inputs
    return [
        {
            "id": 0,
            "question": "What is the official website for the Eiffel Tower? Please answer with a URL only.",
            "gold": "https://www.toureiffel.paris/",
        },
        {
            "id": 1,
            "question": "Who is the current CEO of Amazon? Provide the name only.",
            "gold": "Andy Jassy",
        },
    ]


def simple_normalize(text: str) -> str:
    return (text or "").strip().lower()


def compute_accuracy(predictions: List[str], golds: List[str]) -> float:
    if not predictions:
        return 0.0
    correct = 0
    for pred, gold in zip(predictions, golds):
        if simple_normalize(pred) == simple_normalize(gold):
            correct += 1
    return correct / len(predictions)


async def run_eval() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Please set it in your environment or .env file.")

    tokenizer = None
    if AutoTokenizer is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained("./local_tokenizer")
        except Exception:
            tokenizer = None

    engine_config = {
        "agent_class": StrandsAgentWrapper,
        "env_class": StrandsEnv,
        "agent_args": {},
        "env_args": {},
        "engine_name": "openai",
        "tokenizer": tokenizer,
        "sampling_params": {
            "model": "gpt-4o-mini",
            "temperature": 0.2,
        },
        "rollout_engine_args": {
            "api_key": api_key,
        },
        "n_parallel_agents": 1,
        "max_steps": 3,
        "max_response_length": 500,
        "max_prompt_length": 2000,
    }

    engine = AsyncAgentExecutionEngine(**engine_config)

    tasks = load_tasks()
    # rLLM expects tasks where env.from_dict can map fields -> initial prompt;
    # we pass question as-is and keep id for tracking.
    exec_tasks = [{"id": t["id"], "question": t["question"]} for t in tasks]

    results = await engine.execute_tasks(exec_tasks)

    preds: List[str] = []
    golds: List[str] = []

    for task, res in zip(tasks, results):
        # Best-effort extraction: take the last model response captured in trajectory
        final_response = None
        if res.steps:
            # find last step that has model_response if available
            for step in reversed(res.steps):
                if hasattr(step, "model_response") and step.model_response:
                    final_response = step.model_response
                    break
        if final_response is None:
            # fallback to action if present
            final_response = getattr(res, "action", None) or ""

        preds.append(str(final_response))
        golds.append(task["gold"])

    acc = compute_accuracy(preds, golds)

    print("Results (toy BrowserComp):")
    for t, p, g in zip(tasks, preds, golds):
        print(f"- id={t['id']}\n  Q: {t['question']}\n  Pred: {p}\n  Gold: {g}")
    print(f"\nAccuracy: {acc:.3f} ({sum(simple_normalize(p)==simple_normalize(g) for p,g in zip(preds,golds))}/{len(golds)})")


def main() -> None:
    asyncio.run(run_eval())


if __name__ == "__main__":
    main()
