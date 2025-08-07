import sys
import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine
from rllm_workflow.strands_agent_wrapper import StrandsAgentWrapper
from rllm_workflow.strands_env import StrandsEnv
from eval.browser_env import BrowserEnv
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()

@dataclass
class BrowserCompTask:
    """Represents a BrowserComp benchmark task."""
    id: str
    question: str
    expected_answer: Optional[str] = None
    difficulty: str = "medium"
    category: str = "general"

@dataclass
class EvaluationResult:
    """Represents the evaluation result for a single task."""
    task_id: str
    question: str
    agent_response: str
    expected_answer: Optional[str]
    is_correct: bool
    response_time: float
    steps_taken: int
    reward: float

class BrowserCompEvaluator:
    """Evaluator for BrowserComp benchmark tasks."""
    
    def __init__(self, max_tasks: int = 10):
        self.max_tasks = max_tasks
        self.results: List[EvaluationResult] = []
        
    def load_tasks(self) -> List[BrowserCompTask]:
        """
        Load BrowserComp benchmark tasks.
        
        For now, we'll use a curated set of web search tasks.
        In a real implementation, you would load from the actual BrowserComp dataset.
        """
        tasks = [
            BrowserCompTask(
                id="browser_001",
                question="What is the current population of Tokyo, Japan?",
                expected_answer="approximately 37 million",
                difficulty="easy",
                category="geography"
            ),
            BrowserCompTask(
                id="browser_002", 
                question="Who is the CEO of Microsoft as of 2024?",
                expected_answer="Satya Nadella",
                difficulty="easy",
                category="business"
            ),
            BrowserCompTask(
                id="browser_003",
                question="What is the latest version of Python programming language?",
                expected_answer="Python 3.12",
                difficulty="medium", 
                category="technology"
            ),
            BrowserCompTask(
                id="browser_004",
                question="What are the main ingredients in a traditional margherita pizza?",
                expected_answer="tomato sauce, mozzarella cheese, basil",
                difficulty="easy",
                category="food"
            ),
            BrowserCompTask(
                id="browser_005",
                question="What is the capital of Australia?",
                expected_answer="Canberra",
                difficulty="easy",
                category="geography"
            ),
            BrowserCompTask(
                id="browser_006",
                question="What year was the first iPhone released?",
                expected_answer="2007",
                difficulty="medium",
                category="technology"
            ),
            BrowserCompTask(
                id="browser_007",
                question="Who wrote the novel 'Pride and Prejudice'?",
                expected_answer="Jane Austen",
                difficulty="easy",
                category="literature"
            ),
            BrowserCompTask(
                id="browser_008",
                question="What is the chemical symbol for gold?",
                expected_answer="Au",
                difficulty="easy",
                category="science"
            ),
            BrowserCompTask(
                id="browser_009",
                question="What is the largest planet in our solar system?",
                expected_answer="Jupiter",
                difficulty="easy",
                category="science"
            ),
            BrowserCompTask(
                id="browser_010",
                question="What is the main programming language used for Android development?",
                expected_answer="Java or Kotlin",
                difficulty="medium",
                category="technology"
            )
        ]
        
        return tasks[:self.max_tasks]
    
    def create_browser_env(self, task: BrowserCompTask) -> StrandsEnv:
        """Create a specialized environment for browser tasks."""
        env = StrandsEnv()
        env.current_prompt = task.question
        env.max_steps = 5  # Allow more steps for complex web searches
        return env
    
    def evaluate_response(self, task: BrowserCompTask, response: str) -> bool:
        """
        Evaluate if the agent's response is correct.
        
        This is a simplified evaluation. In practice, you might use:
        - Semantic similarity (e.g., using embeddings)
        - Keyword matching
        - Human evaluation
        - Automated fact-checking APIs
        """
        if not task.expected_answer:
            return True  # No expected answer provided, assume correct
        
        response_lower = response.lower()
        expected_lower = task.expected_answer.lower()
        
        # Simple keyword matching
        expected_keywords = expected_lower.split()
        matched_keywords = sum(1 for keyword in expected_keywords if keyword in response_lower)
        
        # Consider correct if at least 50% of expected keywords are present
        return matched_keywords >= len(expected_keywords) * 0.5
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute evaluation metrics from results."""
        if not self.results:
            return {}
        
        total_tasks = len(self.results)
        correct_answers = sum(1 for result in self.results if result.is_correct)
        accuracy = correct_answers / total_tasks
        
        avg_response_time = sum(result.response_time for result in self.results) / total_tasks
        avg_steps = sum(result.steps_taken for result in self.results) / total_tasks
        avg_reward = sum(result.reward for result in self.results) / total_tasks
        
        # Accuracy by difficulty
        difficulty_accuracy = {}
        for difficulty in ["easy", "medium", "hard"]:
            difficulty_tasks = [r for r in self.results if r.task_id in [t.id for t in self.load_tasks() if t.difficulty == difficulty]]
            if difficulty_tasks:
                difficulty_accuracy[difficulty] = sum(1 for r in difficulty_tasks if r.is_correct) / len(difficulty_tasks)
        
        return {
            "total_tasks": total_tasks,
            "accuracy": accuracy,
            "correct_answers": correct_answers,
            "avg_response_time": avg_response_time,
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "difficulty_accuracy": difficulty_accuracy
        }

async def run_browsercomp_evaluation():
    """Run the BrowserComp benchmark evaluation."""
    print("🚀 Starting BrowserComp Benchmark Evaluation")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
        return
    
    # Initialize evaluator
    evaluator = BrowserCompEvaluator(max_tasks=1)  # Only run 1 task for debugging
    tasks = evaluator.load_tasks()
    
    print(f"📋 Loaded {len(tasks)} BrowserComp tasks")
    
    # Setup engine
    tokenizer = AutoTokenizer.from_pretrained("./local_tokenizer")
    
    engine_config = {
        "agent_class": StrandsAgentWrapper,
        "env_class": BrowserEnv,  # Use specialized browser environment
        "agent_args": {},
        "env_args": {},
        "engine_name": "openai",
        "tokenizer": tokenizer,
        "sampling_params": {
            "model": "gpt-4o-mini",
            "temperature": 0.3,  # Lower temperature for more factual responses
        },
        "rollout_engine_args": {
            "api_key": api_key,
        },
        "n_parallel_agents": 1,
        "max_steps": 5,
        "max_response_length": 1000,
        "max_prompt_length": 2000,
    }
    
    print("🔧 Initializing evaluation engine...")
    engine = AsyncAgentExecutionEngine(**engine_config)
    
    # Run evaluation
    print("\n📊 Running evaluation tasks...")
    print("-" * 50)
    
    for i, task in enumerate(tasks, 1):
        print(f"\n🔄 Task {i}/{len(tasks)}: {task.question}")
        
        start_time = time.time()
        
        # Create task for engine
        engine_task = {
            "id": task.id,
            "prompt": f"Please answer the following question accurately: {task.question}"
        }
        
        # Set the task in the environment
        if hasattr(engine, 'env') and hasattr(engine.env, 'set_task'):
            engine.env.set_task(engine_task)
        
        try:
            # Execute task
            result = await engine.execute_tasks([engine_task])
            result = result[0]  # Get first (and only) result
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Extract agent response from the trajectory
            agent_response = ""
            
            # Debug: Print the result structure
            print(f"DEBUG: Result type: {type(result)}")
            print(f"DEBUG: Result has trajectory: {hasattr(result, 'trajectory')}")
            print(f"DEBUG: Result has steps: {hasattr(result, 'steps')}")
            
            if hasattr(result, 'trajectory') and result.trajectory and result.trajectory.steps:
                print(f"DEBUG: Trajectory steps: {len(result.trajectory.steps)}")
                # Get the last action from the trajectory
                last_step = result.trajectory.steps[-1]
                print(f"DEBUG: Last step action: {repr(last_step.action)}")
                if hasattr(last_step, 'action') and last_step.action:
                    agent_response = str(last_step.action)
            elif result.steps:
                print(f"DEBUG: Direct steps: {len(result.steps)}")
                # Fallback to steps if trajectory is not available
                last_step = result.steps[-1]
                print(f"DEBUG: Last step action: {repr(last_step.action)}")
                print(f"DEBUG: Last step model_response: {repr(last_step.model_response)}")
                if hasattr(last_step, 'model_response') and last_step.model_response:
                    agent_response = last_step.model_response
                elif hasattr(last_step, 'action') and last_step.action:
                    agent_response = str(last_step.action)
            
            if not agent_response:
                print(f"DEBUG: No response found in trajectory/steps")
                # Try to access the result directly
                if hasattr(result, 'response'):
                    agent_response = str(result.response)
                    print(f"DEBUG: Found result.response: {agent_response}")
                elif hasattr(result, 'answer'):
                    agent_response = str(result.answer)
                    print(f"DEBUG: Found result.answer: {agent_response}")
            
            print(f"DEBUG: Final agent_response: {repr(agent_response)}")
            
            # Evaluate response
            is_correct = evaluator.evaluate_response(task, agent_response)
            
            # Create evaluation result
            eval_result = EvaluationResult(
                task_id=task.id,
                question=task.question,
                agent_response=agent_response,
                expected_answer=task.expected_answer,
                is_correct=is_correct,
                response_time=response_time,
                steps_taken=len(result.steps),
                reward=result.reward
            )
            
            evaluator.results.append(eval_result)
            
            # Print result
            status = "✅" if is_correct else "❌"
            print(f"{status} Response: {agent_response[:100]}{'...' if len(agent_response) > 100 else ''}")
            print(f"   Time: {response_time:.2f}s | Steps: {len(result.steps)} | Reward: {result.reward:.2f}")
            
        except Exception as e:
            print(f"❌ Error on task {task.id}: {e}")
            continue
    
    # Compute and display final metrics
    print("\n" + "=" * 50)
    print("📈 EVALUATION RESULTS")
    print("=" * 50)
    
    metrics = evaluator.compute_metrics()
    
    print(f"Total Tasks: {metrics.get('total_tasks', 0)}")
    print(f"Accuracy: {metrics.get('accuracy', 0):.2%}")
    print(f"Correct Answers: {metrics.get('correct_answers', 0)}/{metrics.get('total_tasks', 0)}")
    print(f"Average Response Time: {metrics.get('avg_response_time', 0):.2f}s")
    print(f"Average Steps: {metrics.get('avg_steps', 0):.1f}")
    print(f"Average Reward: {metrics.get('avg_reward', 0):.2f}")
    
    if metrics.get('difficulty_accuracy'):
        print("\nAccuracy by Difficulty:")
        for difficulty, acc in metrics['difficulty_accuracy'].items():
            print(f"  {difficulty.capitalize()}: {acc:.2%}")
    
    # Save detailed results
    results_file = "browsercomp_results.json"
    detailed_results = {
        "timestamp": time.time(),
        "metrics": metrics,
        "results": [
            {
                "task_id": r.task_id,
                "question": r.question,
                "agent_response": r.agent_response,
                "expected_answer": r.expected_answer,
                "is_correct": r.is_correct,
                "response_time": r.response_time,
                "steps_taken": r.steps_taken,
                "reward": r.reward
            }
            for r in evaluator.results
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    print("✅ BrowserComp evaluation completed!")

def main():
    """Main entry point for BrowserComp evaluation."""
    asyncio.run(run_browsercomp_evaluation())

if __name__ == "__main__":
    main()
