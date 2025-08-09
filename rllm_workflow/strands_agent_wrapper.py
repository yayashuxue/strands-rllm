from rllm.agents.agent import BaseAgent, Trajectory, Step, Action
from strands_agent.agent import build_agent

class StrandsAgentWrapper(BaseAgent):
    def __init__(self, **kwargs):
        self.agent = build_agent()
        self._trajectory = Trajectory()
        self._chat_history = []
        self.reset()

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return self._chat_history

    def update_from_env(self, observation: any, reward: float, done: bool, info: dict, **kwargs):
        if isinstance(observation, dict) and "observation" in observation:
            user_message = observation["observation"]
            self._chat_history.append({"role": "user", "content": user_message})

        step = Step(observation=observation, reward=reward, done=done, info=info)
        self._trajectory.steps.append(step)

    def update_from_model(self, response: str, **kwargs) -> Action:
        # Instead of just using the response from rLLM's model,
        # use the Strands agent to generate the actual response
        if self._chat_history:
            # Get the last user message to pass to Strands agent
            last_user_message = None
            for msg in reversed(self._chat_history):
                if msg["role"] == "user":
                    last_user_message = msg["content"]
                    break
            
            if last_user_message:
                try:
                    # Call the actual Strands agent
                    strands_response = self.agent(last_user_message)
                    response = strands_response
                except Exception as e:
                    print(f"Error calling Strands agent: {e}")
                    # Fall back to the original response
                    pass
        
        self._chat_history.append({"role": "assistant", "content": response})
        # --- FIX ---
        # Get the current step and store the model response in it.
        if self._trajectory.steps:
            current_step = self._trajectory.steps[-1]
            current_step.model_response = response
        # ----------
        return Action(action=response)

    def reset(self):
        self._trajectory = Trajectory()
        self._chat_history = [{"role": "system", "content": "You are a helpful assistant."}]
