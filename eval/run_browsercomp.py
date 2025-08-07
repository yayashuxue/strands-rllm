from __future__ import annotations
from typing import List, Dict, Any, Tuple
import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
import re

from dotenv import load_dotenv

# Ensure project root is on sys.path so we can import rllm_workflow when running directly
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore

from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine
from rllm_workflow.strands_agent_wrapper import StrandsAgentWrapper
from rllm_workflow.strands_env import StrandsEnv


"""
Run BrowseComp evaluation with the Strands agent via rLLM.

Usage example:
  python eval/run_browsercomp.py \
    --data /path/to/browsecomp.jsonl \
    --limit 100

DATA FORMAT: Each line (JSONL) or element (JSON array) should contain at least:
  {"id": int|str, "question": str, "answer": str}

Scoring: normalized exact match (case/punct/space insensitive; URL canonicalization).
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Strands+rLLM on BrowseComp")
    parser.add_argument("--data", type=str, default=os.getenv("BROWSECOMP_PATH"), help="Path to BrowseComp JSON or JSONL")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of tasks to evaluate")
    parser.add_argument("--max_steps", type=int, default=3, help="Max dialogue steps per task")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name to use via rLLM")
    return parser.parse_args()


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of objects")
    return data


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_tasks(path_str: str, limit: int | None = None) -> List[Dict[str, Any]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    data = _load_jsonl(path) if path.suffix.lower() == ".jsonl" else _load_json(path)

    tasks: List[Dict[str, Any]] = []
    for i, row in enumerate(data):
        q = row.get("question") or row.get("prompt") or row.get("query")
        a = row.get("answer") or row.get("gold") or row.get("reference")
        tid = row.get("id", i)
        if not q or not a:
            continue
        tasks.append({"id": tid, "question": str(q), "gold": str(a)})
        if limit is not None and len(tasks) >= limit:
            break
    if not tasks:
        raise ValueError("No valid tasks loaded from dataset")
    return tasks


def _strip_url(url: str) -> str:
    # Basic URL canonicalization: lower-case scheme+host, strip trailing slash, remove http(s) prefix
    url = url.strip()
    # Remove surrounding quotes
    if (url.startswith("\"") and url.endswith("\"")) or (url.startswith("'") and url.endswith("'")):
        url = url[1:-1]
    url = url.strip()
    # Remove protocol
    url = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
    # Lowercase host
    parts = url.split("/", 1)
    parts[0] = parts[0].lower()
    url = "/".join(parts)
    # Remove trailing slash
    url = url[:-1] if url.endswith("/") else url
    return url


def simple_normalize(text: str) -> str:
    if text is None:
        return ""
    txt = str(text).strip()
    # If appears like a URL, canonicalize
    if re.match(r"^https?://", txt, flags=re.IGNORECASE) or "." in txt:
        # Heuristic: if it looks like a domain path, try URL normalization
        return _strip_url(txt)
    # Else general normalization
    txt = txt.lower()
    # Remove surrounding quotes
    if (txt.startswith("\"") and txt.endswith("\"")) or (txt.startswith("'") and txt.endswith("'")):
        txt = txt[1:-1]
    # Remove punctuation and excessive whitespace
    txt = re.sub(r"[\s\p{P}]+", " ", txt, flags=re.UNICODE)
    # Fallback for Python's lack of \p{P}: remove common punctuation
    txt = re.sub(r"[\.,;:!?\-—'\"\(\)\[\]\{\}]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def compute_accuracy(predictions: List[str], golds: List[str]) -> Tuple[float, int, int]:
    correct = 0
    for pred, gold in zip(predictions, golds):
        if simple_normalize(pred) == simple_normalize(gold):
            correct += 1
    total = len(golds)
    acc = (correct / total) if total else 0.0
    return acc, correct, total


async def run_eval(data_path: str, limit: int | None, max_steps: int, model: str, temperature: float) -> None:
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
            "model": model,
            "temperature": temperature,
        },
        "rollout_engine_args": {
            "api_key": api_key,
        },
        "n_parallel_agents": 1,
        "max_steps": max_steps,
        "max_response_length": 500,
        "max_prompt_length": 2000,
    }

    engine = AsyncAgentExecutionEngine(**engine_config)

    tasks = load_tasks(data_path, limit=limit)
    exec_tasks = [{"id": t["id"], "question": t["question"], "max_steps": max_steps} for t in tasks]

    results = await engine.execute_tasks(exec_tasks)

    preds: List[str] = []
    golds: List[str] = []

    for task, res in zip(tasks, results):
        final_response = None
        if getattr(res, "steps", None):
            for step in reversed(res.steps):
                if hasattr(step, "model_response") and step.model_response:
                    final_response = step.model_response
                    break
        if final_response is None:
            final_response = getattr(res, "action", None) or ""

        preds.append(str(final_response))
        golds.append(task["gold"])

    acc, num_correct, num_total = compute_accuracy(preds, golds)

    print("BrowseComp results:")
    print(f"Accuracy: {acc:.4f} ({num_correct}/{num_total})")

    # Print a few examples for inspection
    for t, p in list(zip(tasks, preds))[:5]:
        print(f"- id={t['id']}\n  Q: {t['question'][:200]}\n  Pred: {p[:200]}\n  Gold: {t['gold']}")


def main() -> None:
    args = parse_args()
    if not args.data:
        print("--data is required or set BROWSECOMP_PATH env var", file=sys.stderr)
        sys.exit(2)
    asyncio.run(run_eval(args.data, args.limit, args.max_steps, args.model, args.temperature))


if __name__ == "__main__":
    main()
