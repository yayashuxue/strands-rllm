import sys
import os
from pathlib import Path
import asyncio

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine
from rllm_workflow.strands_agent_wrapper import StrandsAgentWrapper
from rllm_workflow.strands_env import StrandsEnv
from transformers import AutoTokenizer

# Load environment variables from .env file
load_dotenv()


def main() -> None:
    """
    Build the Strands agent and run rLLM workflow using OpenAI API (or LiteLLM proxy via base_url).
    """
    print("--- Setting up rLLM Workflow (OpenAI-compatible) ---")

    # Read model/provider from env; works with OpenAI directly or LiteLLM proxy models
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    base_url = os.getenv("LLM_BASE_URL")  # e.g., http://localhost:4000/v1 when using LiteLLM proxy

    # API key: pass through to OpenAI or LiteLLM proxy
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY or LLM_API_KEY not found in environment.")
        return

    print(f"✅ Using model: {model_name}")
    if base_url:
        print(f"✅ Using base_url: {base_url}")

    tokenizer = AutoTokenizer.from_pretrained("./local_tokenizer")
    print("✅ Loaded tokenizer")

    engine_config = {
        "agent_class": StrandsAgentWrapper,
        "env_class": StrandsEnv,
        "agent_args": {},
        "env_args": {},
        # Keep engine OpenAI-compatible; route via LiteLLM if base_url is set
        "engine_name": os.getenv("ENGINE_NAME", "openai"),
        "tokenizer": tokenizer,
        "sampling_params": {
            "model": model_name,
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
        },
        "rollout_engine_args": {
            "api_key": api_key,
            **({"base_url": base_url} if base_url else {}),
        },
        "n_parallel_agents": int(os.getenv("N_PARALLEL_AGENTS", "1")),
        "max_steps": int(os.getenv("MAX_STEPS", "3")),
        "max_response_length": int(os.getenv("MAX_RESPONSE_LENGTH", "500")),
        "max_prompt_length": int(os.getenv("MAX_PROMPT_LENGTH", "2000")),
    }

    print("Initializing AsyncAgentExecutionEngine...")
    engine = AsyncAgentExecutionEngine(**engine_config)

    tasks = [{"id": 0, "prompt": "Hello, who are you?"}]

    print("Executing tasks...")
    try:
        results = asyncio.run(engine.execute_tasks(tasks))
        print("\n--- Workflow Results ---")
        for res in results:
            print(f"Task ID: {res.task.get('id')}")
            print(f"Final Reward: {res.reward}")
            print(f"Number of steps: {len(res.steps)}")

            if res.steps:
                for i, step in enumerate(res.steps):
                    print(f"Step {i}:")
                    print(f"  Observation: {step.observation}")
                    print(f"  Reward: {step.reward}")
                    print(f"  Done: {step.done}")
                    if hasattr(step, 'model_response') and step.model_response:
                        print(f"  Model Response: {step.model_response}")
                    else:
                        print(f"  Model Response: (empty)")
                    print()
        print("✅ rLLM workflow completed successfully!")

    except Exception as e:
        print(f"❌ An error occurred during task execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 