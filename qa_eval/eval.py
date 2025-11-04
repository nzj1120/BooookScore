"""End-to-end evaluation logic for QA-based summary assessment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple, TypeVar

from .api import ModelSpec, build_client
from .prompts import load_prompt
from .report import BookQAReport, QAInteraction
from .utils import Judgement, QARecord, extract_json_array, extract_json_object, mean


@dataclass
class GenerationConfig:
    model: ModelSpec
    max_tokens: int = 2048
    temperature: float = 0.2


@dataclass
class JudgeConfig:
    model: ModelSpec
    max_tokens: int = 512
    temperature: float = 0.0


@dataclass
class EvaluationConfig:
    question_count: int = 5
    max_retries: int = 3
    generation: GenerationConfig | None = None
    judge: JudgeConfig | None = None

    def ensure_clients(self) -> Tuple[GenerationConfig, JudgeConfig]:
        if self.generation is None:
            raise ValueError("Generation configuration is required.")
        if self.judge is None:
            self.judge = JudgeConfig(model=self.generation.model, max_tokens=512, temperature=0.0)
        return self.generation, self.judge


_GENERATE_PROMPT = None
_VERIFY_PROMPT = None


_T = TypeVar("_T")


def _run_with_retries(func: Callable[[], _T], description: str, attempts: int) -> _T:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except ValueError as exc:
            last_error = exc
            print(
                f"[warning] {description} attempt {attempt}/{attempts} failed: {exc}"
            )
    if last_error is not None:
        raise ValueError(
            f"{description} failed after {attempts} attempts: {last_error}"
        ) from last_error
    raise ValueError(f"{description} failed with no attempts executed.")


def _prompt_generate() -> str:
    global _GENERATE_PROMPT
    if _GENERATE_PROMPT is None:
        _GENERATE_PROMPT = load_prompt("generate_questions.txt")
    return _GENERATE_PROMPT


def _prompt_verify() -> str:
    global _VERIFY_PROMPT
    if _VERIFY_PROMPT is None:
        _VERIFY_PROMPT = load_prompt("verify_answer.txt")
    return _VERIFY_PROMPT


def _format_generate_prompt(text: str, text_type: str, question_count: int) -> str:
    return _prompt_generate().format(text=text, text_type=text_type, question_count=question_count)


def _format_verify_prompt(question: str, answer: str, context: str) -> str:
    return _prompt_verify().format(question=question, answer=answer, context=context)


def generate_questions(text: str, text_type: str, config: EvaluationConfig, client) -> List[QARecord]:
    prompt = _format_generate_prompt(text=text, text_type=text_type, question_count=config.question_count)
    generation_cfg, _ = config.ensure_clients()

    def _request() -> List[dict]:
        response = client.complete(
            prompt,
            max_tokens=generation_cfg.max_tokens,
            temperature=generation_cfg.temperature,
        )
        return extract_json_array(response)

    raw_items = _run_with_retries(
        _request,
        description=f"question generation for {text_type}",
        attempts=max(1, config.max_retries),
    )
    questions: List[QARecord] = []
    for payload in raw_items[: config.question_count]:
        try:
            questions.append(QARecord.from_payload(payload))
        except ValueError as exc:
            print(f"Skipping malformed QA pair: {exc}")
    return questions


def judge_answer(question: QARecord, context: str, config: EvaluationConfig, client) -> Judgement:
    _, judge_cfg = config.ensure_clients()
    prompt = _format_verify_prompt(question=question.question, answer=question.answer, context=context)

    def _request() -> Judgement:
        response = client.complete(
            prompt,
            max_tokens=judge_cfg.max_tokens,
            temperature=judge_cfg.temperature,
        )
        payload = extract_json_object(response)
        return Judgement.from_payload(payload)

    return _run_with_retries(
        _request,
        description="answer verification",
        attempts=max(1, config.max_retries),
    )


def evaluate_book(
    book_id: str,
    source_text: str,
    summary_text: str,
    config: EvaluationConfig,
    *,
    generation_client=None,
    judge_client=None,
) -> BookQAReport:
    generation_cfg, judge_cfg = config.ensure_clients()
    generation_client = generation_client or build_client(generation_cfg.model)
    judge_client = judge_client or build_client(judge_cfg.model)

    summary_questions = generate_questions(summary_text, "summary", config, generation_client)
    source_questions = generate_questions(source_text, "source", config, generation_client)

    hallucination_results: List[QAInteraction] = []
    coverage_results: List[QAInteraction] = []

    for qa in summary_questions:
        judgement = judge_answer(qa, source_text, config, judge_client)
        hallucination_results.append(QAInteraction(question=qa, judgement=judgement, origin="summary"))

    for qa in source_questions:
        judgement = judge_answer(qa, summary_text, config, judge_client)
        coverage_results.append(QAInteraction(question=qa, judgement=judgement, origin="source"))

    hallucination_score = mean(entry.judgement.is_correct for entry in hallucination_results) if hallucination_results else 0.0
    coverage_score = mean(entry.judgement.is_correct for entry in coverage_results) if coverage_results else 0.0

    return BookQAReport(
        book_id=book_id,
        requested_question_count=config.question_count,
        summary_question_count=len(summary_questions),
        source_question_count=len(source_questions),
        hallucination_score=hallucination_score,
        coverage_score=coverage_score,
        hallucination_details=hallucination_results,
        coverage_details=coverage_results,
    )
