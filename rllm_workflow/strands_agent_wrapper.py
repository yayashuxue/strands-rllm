from rllm.agents.agent import BaseAgent, Trajectory, Step, Action
from strands_agent.agent import build_agent
import os
try:
    from strands_agent.memory import Mem1Memory  # Optional MEM1 backend
except Exception:
    Mem1Memory = None  # type: ignore

class StrandsAgentWrapper(BaseAgent):
    def __init__(self, memory_backend: str | None = None, **kwargs):
        self.agent = build_agent()
        self._trajectory = Trajectory()
        self._chat_history = []
        backend = (memory_backend or os.getenv("STRANDS_MEMORY", "mem0")).lower()
        self._memory = Mem1Memory() if backend == "mem1" and Mem1Memory else None
        self._verbose = os.getenv("STRANDS_VERBOSE", "0").lower() in ("1", "true")
        self.reset()

    def _normalize_response_text(self, result) -> str:
        if isinstance(result, str):
            return result
        try:
            content = getattr(result, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for piece in content:
                    if isinstance(piece, dict) and "text" in piece:
                        texts.append(str(piece["text"]))
                    else:
                        texts.append(str(piece))
                joined = "\n".join(texts).strip()
                if joined:
                    return joined
        except Exception:
            pass
        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for piece in content:
                    if isinstance(piece, dict) and "text" in piece:
                        texts.append(str(piece["text"]))
                    else:
                        texts.append(str(piece))
                return "\n".join(texts).strip()
            if "text" in result:
                return str(result["text"])
        try:
            return str(result)
        except Exception:
            return ""

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return self._chat_history

    def update_from_env(self, observation: any, reward: float, done: bool, info: dict, **kwargs):
        if isinstance(observation, dict) and "observation" in observation:
            user_message = observation["observation"]
            if self._verbose:
                print(f"[StrandsWrapper] Env -> user: {user_message[:200]}")
            self._chat_history.append({"role": "user", "content": user_message})
            if self._memory and isinstance(user_message, str):
                try:
                    self._memory.update(user_message)
                    if self._verbose:
                        print("[StrandsWrapper] MEM1.update()")
                except Exception as e:
                    if self._verbose:
                        print(f"[StrandsWrapper] MEM1.update() error: {e}")

        self._trajectory.steps.append(Step(observation=observation, reward=reward, done=done, info=info))

    def update_from_model(self, response: str, **kwargs) -> Action:
        enriched = None
        if self._chat_history:
            last_user = next((m["content"] for m in reversed(self._chat_history) if m.get("role") == "user"), None)
            if last_user:
                if self._memory and isinstance(last_user, str):
                    try:
                        recalled = self._memory.recall(last_user)
                        if isinstance(recalled, str) and recalled.strip():
                            enriched = f"[Memory]\n{recalled}\n\n[Query]\n{last_user}"
                            if self._verbose:
                                print("[StrandsWrapper] MEM1.recall() injected")
                    except Exception as e:
                        if self._verbose:
                            print(f"[StrandsWrapper] MEM1.recall() error: {e}")
                if enriched is None:
                    enriched = last_user

        if enriched:
            if self._verbose:
                print(f"[StrandsWrapper] Calling Strands with:\n{enriched[:500]}")
            try:
                strands_resp = self.agent(enriched)
                response = self._normalize_response_text(strands_resp)
                if self._verbose:
                    print(f"[StrandsWrapper] Strands resp: {response[:200]}")
            except Exception as e:
                print(f"Error calling Strands agent: {e}")

        resp_text = response if isinstance(response, str) else self._normalize_response_text(response)
        self._chat_history.append({"role": "assistant", "content": resp_text})
        if self._trajectory.steps:
            self._trajectory.steps[-1].model_response = resp_text
        return Action(action=resp_text)

    def reset(self):
        self._trajectory = Trajectory()
        self._chat_history = [{"role": "system", "content": "You are a helpful assistant."}]
        if self._verbose:
            tool_names = []
            for t in getattr(self.agent, 'tools', []):
                name = getattr(t, 'name', None) or getattr(getattr(t, '__func__', t), '__name__', None) or type(t).__name__
                tool_names.append(name)
            print(f"[StrandsWrapper] Tools: {', '.join(tool_names) or '[]'}")
            print(f"[StrandsWrapper] Memory backend: {'MEM1' if self._memory else 'mem0/default'}")
