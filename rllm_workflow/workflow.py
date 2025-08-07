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
    Build the Strands agent and run rLLM workflow using OpenAI API.
    """
    print("--- Setting up rLLM Workflow for OpenAI ---")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found or not set in .env file.")
        return

    print(f"✅ Found API key: {api_key[:10]}...")

    tokenizer = AutoTokenizer.from_pretrained("./local_tokenizer")
    print("✅ Loaded tokenizer")

    engine_config = {
        "agent_class": StrandsAgentWrapper,
        "env_class": StrandsEnv,
        "agent_args": {},
        "env_args": {},
        "engine_name": "openai",
        "tokenizer": tokenizer,
        "sampling_params": {
            "model": "gpt-4o-mini",  # Use o3-mini - the only model that works with completions API
            "temperature": 0.7,  # Higher temperature for more creative responses
            # Remove max_tokens from here to avoid duplicate parameter error
        },
        "rollout_engine_args": {
            "api_key": api_key,
        },
        "n_parallel_agents": 1,
        "max_steps": 3,  # Allow up to 3 conversation turns
        "max_response_length": 500,
        "max_prompt_length": 2000,
    }

    print("Initializing AsyncAgentExecutionEngine...")
    engine = AsyncAgentExecutionEngine(**engine_config)

    tasks = [{"id": 0, "prompt": "Hello, who are you?"}]

    print("Executing tasks with OpenAI...")
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