from __future__ import annotations

from eval.evaluators.base import BaseEvaluator
from eval.loader import EvaluationBundle


class CriticalActionStrictEvaluator(BaseEvaluator):
    name = "critical_action_strict"

    def evaluate(self, bundle: EvaluationBundle) -> dict:
        critical_actions = list(bundle.task_yaml.get("critical_actions", []))
        successful_entries = [
            {"log_index": index, "entry": entry}
            for index, entry in enumerate(bundle.run_result.get("execution_log", []))
            if entry.get("success") is True
        ]
        matched_log_indices: list[int] = []
        used_positions: set[int] = set()

        for action in critical_actions:
            # `params` semantic: presence/absence on a critical_actions
            # entry encodes the match mode.
            #  - present (including {}): strict -- `entry.params` must
            #    equal `action.params` exactly. This covers node-expanded
            #    entries (loader always populates params from the DAG)
            #    and additional_critical_actions entries that care about
            #    exact call shape.
            #  - absent (key missing): tool-only match, like the loose
            #    evaluator. Used by additional_critical_actions entries
            #    where the interpretation's exact params depend on what
            #    the agent chose (e.g. ambiguous: which candidate email
            #    the agent targeted).
            # See eval/loader.py:_expand_critical_actions for how this
            # gets written.
            check_params = "params" in action
            for position, candidate in enumerate(successful_entries):
                if position in used_positions:
                    continue
                entry = candidate["entry"]
                if entry.get("tool") != action.get("tool"):
                    continue
                if check_params and entry.get("params") != action["params"]:
                    continue
                used_positions.add(position)
                matched_log_indices.append(candidate["log_index"])
                break

        matched_count = len(matched_log_indices)
        total = len(critical_actions)
        pass_value = matched_count == total if bundle.task_type == "act" else matched_count == 0
        return {
            "pass": pass_value,
            "matched_count": matched_count,
            "total_critical_actions": total,
            "matched_log_indices": matched_log_indices,
        }
