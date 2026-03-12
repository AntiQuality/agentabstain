from __future__ import annotations

from collections import Counter

from eval.evaluators.base import BaseEvaluator
from eval.loader import EvaluationBundle


class CriticalActionLooseEvaluator(BaseEvaluator):
    name = "critical_action_loose"

    def evaluate(self, bundle: EvaluationBundle) -> dict:
        critical_actions = list(bundle.task_yaml.get("critical_actions", []))
        successful_entries = [
            {"log_index": index, "entry": entry}
            for index, entry in enumerate(bundle.run_result.get("execution_log", []))
            if entry.get("success") is True
        ]
        required_by_tool = Counter(str(action.get("tool")) for action in critical_actions)
        matched_log_indices: list[int] = []

        for candidate in successful_entries:
            tool_name = str(candidate["entry"].get("tool"))
            if required_by_tool[tool_name] <= 0:
                continue
            required_by_tool[tool_name] -= 1
            matched_log_indices.append(candidate["log_index"])

        total = len(critical_actions)
        matched_count = len(matched_log_indices)
        pass_value = matched_count == total if bundle.task_type == "act" else matched_count == 0
        return {
            "pass": pass_value,
            "matched_count": matched_count,
            "total_critical_actions": total,
            "matched_log_indices": matched_log_indices,
        }
