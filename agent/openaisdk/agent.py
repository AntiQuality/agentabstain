from __future__ import annotations

from pathlib import Path

from src.runtime.openaisdk import run_openai_task
from src.types.BaseAgent import BaseAgent, TaskRunResult


class OpenAISDKAgent(BaseAgent):
    def __init__(self, model: str, temperature: float, max_turns: int, results_root: str | Path):
        super().__init__(model=model, temperature=temperature, max_turns=max_turns, results_root=results_root)
        self.repo_root = Path(__file__).resolve().parents[2]

    async def arun(self, category: str, task_id: str | int, task_type: str) -> TaskRunResult:
        bundle = self.load_task_bundle(category, task_id, task_type)
        return await run_openai_task(self, bundle, self.repo_root)
