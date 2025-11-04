"""Reporting helpers for QA-based evaluation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List

from .utils import Judgement, QARecord


@dataclass
class QAInteraction:
    """Stores a generated question and the evaluation against another text."""

    question: QARecord
    judgement: Judgement
    origin: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "origin": self.origin,
            "question": self.question.question,
            "expected_answer": self.question.answer,
            "verdict": self.judgement.verdict,
            "justification": self.judgement.justification,
            "answer_from_context": self.judgement.answer_from_context,
        }


@dataclass
class BookQAReport:
    book_id: str
    requested_question_count: int
    summary_question_count: int
    source_question_count: int
    hallucination_score: float
    coverage_score: float
    hallucination_details: List[QAInteraction] = field(default_factory=list)
    coverage_details: List[QAInteraction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "book_id": self.book_id,
            "requested_question_count": self.requested_question_count,
            "summary_question_count": self.summary_question_count,
            "source_question_count": self.source_question_count,
            "hallucination_score": self.hallucination_score,
            "coverage_score": self.coverage_score,
            "summary_to_source": [entry.to_dict() for entry in self.hallucination_details],
            "source_to_summary": [entry.to_dict() for entry in self.coverage_details],
        }


def write_report(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
