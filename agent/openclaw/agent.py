from __future__ import annotations

from pathlib import Path

from src.runtime.openclaw import run_openclaw_task
from src.types.BaseAgent import BaseAgent, TaskRunResult


class OpenClawAgent(BaseAgent):
    """Drives the OpenClaw CLI as a thin agent harness over Bedrock.

    OpenClaw spawns the per-task MCP server itself (via the global
    `mcp.servers` registry); we read final state from the dump file the
    server writes when RUNTIME_STATE_DUMP_PATH is set.
    """

    def __init__(self, model: str, temperature: float, max_turns: int, results_root: str | Path):
        super().__init__(model=model, temperature=temperature, max_turns=max_turns, results_root=results_root)
        self.repo_root = Path(__file__).resolve().parents[2]

    async def arun(self, category: str, task_id: str | int, task_type: str) -> TaskRunResult:
        bundle = self.load_task_bundle(category, task_id, task_type)
        return await run_openclaw_task(self, bundle, self.repo_root)
