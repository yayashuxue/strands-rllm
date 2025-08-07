"""
Browser environment for BrowserComp benchmark tasks.

This environment simulates web search interactions for evaluating
the agent's ability to answer factual questions.
"""

from rllm.environments.base.base_env import BaseEnv
from typing import Dict, Any, Optional
import random

class BrowserEnv(BaseEnv):
    """Environment that simulates web search for BrowserComp tasks."""
    
    def __init__(self, **kwargs):
        self.current_task = None
        self.conversation_step = 0
        self.max_steps = 5
        self.search_results = []
        self.current_question = ""
        
    @staticmethod
    def from_dict(data: dict):
        return BrowserEnv(**data)
    
    def set_task(self, task: Dict[str, Any]):
        """Set the current task for the environment."""
        self.current_task = task
        self.current_question = task.get("prompt", "")
        self.conversation_step = 0
        self.search_results = []
        
        # Simulate some search results based on the question
        self._generate_search_results()
    
    def _generate_search_results(self):
        """Generate simulated search results for the current question."""
        # This is a simplified simulation. In a real implementation,
        # you might integrate with actual web search APIs or use
        # pre-collected search results.
        
        question_lower = self.current_question.lower()
        
        # Simulate different types of search results based on question content
        if "population" in question_lower and "tokyo" in question_lower:
            self.search_results = [
                "Tokyo population: approximately 37 million people as of 2024",
                "Tokyo is the most populous metropolitan area in the world",
                "The Greater Tokyo Area has over 37 million residents"
            ]
        elif "ceo" in question_lower and "microsoft" in question_lower:
            self.search_results = [
                "Satya Nadella is the CEO of Microsoft since 2014",
                "Microsoft CEO: Satya Nadella leads the company",
                "Satya Nadella became Microsoft's CEO in 2014"
            ]
        elif "python" in question_lower and "version" in question_lower:
            self.search_results = [
                "Python 3.12 is the latest stable version",
                "Current Python version: 3.12.x",
                "Python 3.12 was released in October 2023"
            ]
        else:
            # Generic search results
            self.search_results = [
                "Search results for your query...",
                "Here is some relevant information...",
                "Based on recent data..."
            ]
    
    def step(self, action):
        """
        Process the agent's action and return the next observation.
        
        Args:
            action: The agent's response/action
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        self.conversation_step += 1
        
        # Calculate reward based on action quality
        reward = self._calculate_reward(action)
        
        # Check if conversation should end
        done = self.conversation_step >= self.max_steps
        
        # Generate next observation based on conversation flow
        observation = self._generate_observation()
        
        info = {
            "step": self.conversation_step,
            "search_results": self.search_results,
            "question": self.current_question
        }
        
        return {"observation": observation}, reward, done, info
    
    def _calculate_reward(self, action) -> float:
        """Calculate reward based on the agent's action."""
        if not action or len(str(action).strip()) == 0:
            return 0.0
        
        # Base reward for providing a response
        reward = 1.0
        
        # Additional reward for longer, more detailed responses
        action_length = len(str(action))
        if action_length > 50:
            reward += 0.5
        if action_length > 100:
            reward += 0.5
        
        # Reward for using search results (simplified check)
        action_lower = str(action).lower()
        for result in self.search_results:
            if any(word in action_lower for word in result.lower().split()[:3]):
                reward += 0.3
                break
        
        return min(reward, 3.0)  # Cap reward at 3.0
    
    def _generate_observation(self) -> str:
        """Generate the next observation for the agent."""
        if self.conversation_step == 1:
            # First step: provide search results
            if self.search_results:
                return f"Search results for '{self.current_question}':\n" + "\n".join(self.search_results)
            else:
                return f"Searching for information about: {self.current_question}"
        
        elif self.conversation_step == 2:
            # Second step: ask for clarification or additional info
            return "Can you provide more specific details or clarify your answer?"
        
        elif self.conversation_step == 3:
            # Third step: provide additional context
            return "Here's some additional context that might be helpful..."
        
        elif self.conversation_step == 4:
            # Fourth step: final prompt
            return "Please provide your final answer based on all the information."
        
        else:
            # Final step
            return "Thank you for your response. Evaluation complete."
    
    def reset(self):
        """Reset the environment for a new conversation."""
        self.conversation_step = 0
        self.current_task = None
        self.current_question = ""
        self.search_results = []
        return {"observation": "Ready for a new browser search task."}, {} 