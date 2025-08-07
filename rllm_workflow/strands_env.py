from rllm.environments.base.base_env import BaseEnv

class StrandsEnv(BaseEnv):
    def __init__(self, initial_prompt: str | None = None, max_steps: int = 3, **kwargs):
        self.current_prompt = initial_prompt
        self.initial_prompt = initial_prompt
        self.conversation_step = 0
        self.max_steps = max_steps

    @staticmethod
    def from_dict(data: dict):
        # Map common task fields to an initial prompt for the environment
        initial_prompt = data.get("prompt") or data.get("question") or data.get("input")
        max_steps = data.get("max_steps", 3)
        return StrandsEnv(initial_prompt=initial_prompt, max_steps=max_steps)

    def step(self, action):
        """
        Process the agent's action and return the next observation.
        
        Args:
            action: The agent's response/action
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        self.conversation_step += 1
        
        # Simple reward logic: give positive reward for any response
        reward = 1.0 if action and len(str(action).strip()) > 0 else 0.0
        
        # Check if conversation should end
        done = self.conversation_step >= self.max_steps
        
        # Create next observation based on the conversation flow
        if self.conversation_step == 1:
            # First follow-up after initial prompt
            observation = "Thank you. Can you expand with specific details or sources?"
        elif self.conversation_step == 2:
            # Second follow-up
            observation = "What would be your final concise answer?"
        else:
            # Final response
            observation = "End of conversation."
            done = True
        
        return {"observation": observation}, reward, done, {}

    def reset(self):
        """Reset the environment for a new conversation."""
        self.conversation_step = 0
        self.current_prompt = self.initial_prompt
        # Use the task-provided prompt if available; otherwise a default greeting
        first_observation = self.current_prompt or "Hello, who are you?"
        return {"observation": first_observation}, {}