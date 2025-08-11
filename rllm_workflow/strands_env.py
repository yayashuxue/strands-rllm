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
        Process the agent's action and advance the dialogue turn.

        Returns:
            tuple: (observation, reward, done, info)
        """
        self.conversation_step += 1

        # Keep reward neutral; scoring is done externally in the runner
        reward = 0.0

        # Done when reaching max_steps
        done = self.conversation_step >= self.max_steps

        # Do not inject any additional guidance; let the agent drive the flow
        observation = ""

        return {"observation": observation}, reward, done, {}

    def reset(self):
        """Reset the environment for a new conversation."""
        self.conversation_step = 0
        self.current_prompt = self.initial_prompt
        # Provide the task prompt as the initial observation
        first_observation = self.current_prompt or ""
        return {"observation": first_observation}, {}