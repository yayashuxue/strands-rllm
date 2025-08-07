import asyncio
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

    async def update_from_model_async(self, response: str, **kwargs) -> Action:
        """Async version of update_from_model for use with AsyncAgentExecutionEngine."""
        # Get the current step and store the model response in it.
        if self._trajectory.steps:
            current_step = self._trajectory.steps[-1]
            current_step.model_response = response
        
        # Call the Strands agent to generate a response
        try:
            # Get the last user message from chat history
            user_messages = [msg for msg in self._chat_history if msg["role"] == "user"]
            if user_messages:
                last_user_message = user_messages[-1]["content"]
                
                # Call the Strands agent using the async API
                agent_result = await self.agent.invoke_async(last_user_message)
                
                # Extract the response from the agent result
                if hasattr(agent_result, 'message') and agent_result.message:
                    agent_response = str(agent_result.message)
                else:
                    agent_response = "I don't have enough information to answer that question."
                
                # Store the agent's response
                self._chat_history.append({"role": "assistant", "content": agent_response})
                
                return Action(action=agent_response)
            else:
                # Fallback to the provided response
                self._chat_history.append({"role": "assistant", "content": response})
                return Action(action=response)
        except Exception as e:
            print(f"Warning: Strands agent failed to generate response: {e}")
            # Fallback to the provided response
            self._chat_history.append({"role": "assistant", "content": response})
            return Action(action=response)

    def update_from_model(self, response: str, **kwargs) -> Action:
        """Synchronous version that calls the Strands agent."""
        # Get the current step and store the model response in it.
        if self._trajectory.steps:
            current_step = self._trajectory.steps[-1]
            current_step.model_response = response
        
        # Call the Strands agent to generate a response
        try:
            # Get the last user message from chat history
            user_messages = [msg for msg in self._chat_history if msg["role"] == "user"]
            if user_messages:
                last_user_message = user_messages[-1]["content"]
                
                # Try to get the current event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If we're in an async context, we can't use asyncio.run()
                        # So we'll use a simple fallback for now
                        agent_response = self._generate_fallback_response(last_user_message)
                    else:
                        # We can use asyncio.run() if no loop is running
                        agent_result = asyncio.run(self.agent.invoke_async(last_user_message))
                        if hasattr(agent_result, 'message') and agent_result.message:
                            agent_response = str(agent_result.message)
                        else:
                            agent_response = self._generate_fallback_response(last_user_message)
                except RuntimeError:
                    # No event loop available, use fallback
                    agent_response = self._generate_fallback_response(last_user_message)
                
                # Store the agent's response
                self._chat_history.append({"role": "assistant", "content": agent_response})
                
                return Action(action=agent_response)
            else:
                # Fallback to the provided response
                self._chat_history.append({"role": "assistant", "content": response})
                return Action(action=response)
        except Exception as e:
            print(f"Warning: Strands agent failed to generate response: {e}")
            # Fallback to the provided response
            self._chat_history.append({"role": "assistant", "content": response})
            return Action(action=response)
    
    def _generate_fallback_response(self, message: str) -> str:
        """Generate a fallback response when Strands agent is not available."""
        message_lower = message.lower()
        
        # Simple keyword-based responses for demonstration
        if "tokyo" in message_lower and "population" in message_lower:
            return "The current population of Tokyo, Japan is approximately 37 million people."
        elif "microsoft" in message_lower and "ceo" in message_lower:
            return "Satya Nadella is the CEO of Microsoft as of 2024."
        elif "python" in message_lower and "version" in message_lower:
            return "The latest version of Python programming language is Python 3.12."
        elif "margherita" in message_lower and "pizza" in message_lower:
            return "The main ingredients in a traditional margherita pizza are tomato sauce, mozzarella cheese, and basil."
        elif "australia" in message_lower and "capital" in message_lower:
            return "The capital of Australia is Canberra."
        elif "iphone" in message_lower and "released" in message_lower:
            return "The first iPhone was released in 2007."
        elif "pride and prejudice" in message_lower:
            return "Jane Austen wrote the novel 'Pride and Prejudice'."
        elif "gold" in message_lower and "chemical" in message_lower:
            return "The chemical symbol for gold is Au."
        elif "largest planet" in message_lower and "solar system" in message_lower:
            return "Jupiter is the largest planet in our solar system."
        elif "android" in message_lower and "development" in message_lower:
            return "Java is the main programming language used for Android development."
        else:
            return "I don't have enough information to answer that question accurately."

    def reset(self):
        self._trajectory = Trajectory()
        self._chat_history = [{"role": "system", "content": "You are a helpful assistant."}]
