from __future__ import annotations

from abc import ABC, abstractmethod

from eval.loader import EvaluationBundle


class BaseEvaluator(ABC):
    name: str

    @abstractmethod
    def evaluate(self, bundle: EvaluationBundle) -> dict:
        """Return a serializable metric result for one run."""
