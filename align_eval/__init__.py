"""AlignEval: tools for evaluating summary faithfulness via sentence alignment."""

from .eval import EvaluationConfig, evaluate_alignment, evaluate_corpus

__all__ = ["EvaluationConfig", "evaluate_alignment", "evaluate_corpus"]
