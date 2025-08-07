from rllm.environments.base.base_env import BaseEnv

class StrandsEnv(BaseEnv):
    def __init__(self, **kwargs):
        self.current_prompt = None
        self.conversation_step = 0
        self.max_steps = 3  # Allow up to 3 conversation turns

    @staticmethod
    def from_dict(data: dict):
        return StrandsEnv(**data)

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
            # First response to initial prompt
            observation = "Thank you for your introduction. Can you tell me more about your capabilities?"
        elif self.conversation_step == 2:
            # Second response
            observation = "That's interesting! What's your favorite programming language?"
        else:
            # Final response
            observation = "Great conversation! Have a nice day."
            done = True
        
        return {"observation": observation}, reward, done, {}

    def reset(self):
        """Reset the environment for a new conversation."""
        self.conversation_step = 0
        self.current_prompt = None
        return {"observation": "Hello, who are you?"}, {}