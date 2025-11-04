"""Utility helpers for QA-based evaluation."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
JSON_ARRAY_PATTERN = re.compile(r"\[.*\]", re.DOTALL)


def _strip_trailing(text: str) -> str:
    return text.strip().strip("` ")


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    """Best-effort JSON array extraction from an LLM response."""
    cleaned = _strip_trailing(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    match = JSON_ARRAY_PATTERN.search(cleaned)
    if not match:
        raise ValueError("Failed to locate JSON array in response.")
    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("Extracted JSON is not a list.")
    return data


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = _strip_trailing(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = JSON_OBJECT_PATTERN.search(cleaned)
    if not match:
        raise ValueError("Failed to locate JSON object in response.")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Extracted JSON is not an object.")
    return data


@dataclass
class QARecord:
    question: str
    answer: str

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "QARecord":
        question = str(payload.get("question", "")).strip()
        answer = str(payload.get("answer", "")).strip()
        if not question or not answer:
            raise ValueError("Question and answer must be non-empty.")
        return cls(question=question, answer=answer)


@dataclass
class Judgement:
    verdict: str
    justification: str
    answer_from_context: str

    @property
    def is_correct(self) -> bool:
        return self.verdict.lower().strip() == "correct"

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "Judgement":
        verdict = str(payload.get("verdict", "")).strip()
        justification = str(payload.get("justification", "")).strip()
        answer_from_context = str(payload.get("answer_from_context", "")).strip()
        if verdict.lower() not in {"correct", "incorrect"}:
            raise ValueError("Verdict must be 'correct' or 'incorrect'.")
        return cls(verdict=verdict, justification=justification, answer_from_context=answer_from_context)


def mean(values: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for value in values:
        total += float(value)
        count += 1
    return total / count if count else 0.0
