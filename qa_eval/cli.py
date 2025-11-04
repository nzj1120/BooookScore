"""Command line interface for question-answer based summary evaluation."""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

from tqdm.auto import tqdm

from .api import ModelSpec, build_client
from .eval import EvaluationConfig, GenerationConfig, JudgeConfig, evaluate_book
from .report import BookQAReport, write_report
from .utils import mean


def _load_json(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at {path}")
    return {str(key): str(value) for key, value in data.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate summaries via cross-question answering")
    parser.add_argument("--source_path", required=True, help="Path to original text JSON")
    parser.add_argument("--summary_path", required=True, help="Path to summary JSON")
    parser.add_argument("--output_path", required=True, help="Where to save evaluation report JSON")

    parser.add_argument("--api", choices=["openai", "anthropic", "together"], default="openai")
    parser.add_argument("--api_key", required=True, help="API key string or path to file containing it")
    parser.add_argument("--model", required=True, help="Base model used for question generation and judging")
    parser.add_argument("--generation_model", help="Optional override for question generation model")
    parser.add_argument("--judge_model", help="Optional override for answer verification model")
    parser.add_argument("--base_url", default=None, help="Optional custom base URL for OpenAI-compatible services")

    parser.add_argument("--question_count", type=int, default=5, help="Number of QA pairs to sample from each text")
    parser.add_argument("--max_retries", type=int, default=3, help="Number of times to retry when parsing model output fails")
    parser.add_argument("--generation_max_tokens", type=int, default=2048)
    parser.add_argument("--generation_temperature", type=float, default=0.2)
    parser.add_argument("--judge_max_tokens", type=int, default=512)
    parser.add_argument("--judge_temperature", type=float, default=0.0)

    parser.add_argument("--show_progress", action="store_true", help="Display per-book progress bar")

    return parser


def _build_model_spec(api: str, api_key: str, model: str, base_url: Optional[str]) -> ModelSpec:
    return ModelSpec(api=api, api_key=api_key, model=model, base_url=base_url)


def main(args: Optional[List[str]] = None) -> Dict[str, float]:
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    source_data = _load_json(parsed.source_path)
    summary_data = _load_json(parsed.summary_path)

    missing = sorted(set(summary_data) - set(source_data))
    if missing:
        raise ValueError(f"Missing source text for: {missing}")

    generation_model_name = parsed.generation_model or parsed.model
    judge_model_name = parsed.judge_model or parsed.model

    generation_spec = _build_model_spec(parsed.api, parsed.api_key, generation_model_name, parsed.base_url)
    judge_spec = _build_model_spec(parsed.api, parsed.api_key, judge_model_name, parsed.base_url)

    eval_config = EvaluationConfig(
        question_count=parsed.question_count,
        max_retries=parsed.max_retries,
        generation=GenerationConfig(
            model=generation_spec,
            max_tokens=parsed.generation_max_tokens,
            temperature=parsed.generation_temperature,
        ),
        judge=JudgeConfig(
            model=judge_spec,
            max_tokens=parsed.judge_max_tokens,
            temperature=parsed.judge_temperature,
        ),
    )

    generation_client = build_client(generation_spec)
    judge_client = build_client(judge_spec)

    iterator = summary_data.items()
    if parsed.show_progress:
        iterator = tqdm(iterator, total=len(summary_data), desc="Evaluating QA overlap")

    reports: Dict[str, BookQAReport] = {}
    hallucination_scores = []
    coverage_scores = []

    for book_id, summary_text in iterator:
        source_text = source_data[book_id]
        report = evaluate_book(
            book_id,
            source_text,
            summary_text,
            eval_config,
            generation_client=generation_client,
            judge_client=judge_client,
        )
        reports[book_id] = report
        hallucination_scores.append(report.hallucination_score)
        coverage_scores.append(report.coverage_score)

    macro_metrics = {
        "hallucination_score": mean(hallucination_scores),
        "coverage_score": mean(coverage_scores),
    }

    payload = {
        "config": {
            "api": parsed.api,
            "generation_model": generation_model_name,
            "judge_model": judge_model_name,
            "question_count": parsed.question_count,
            "max_retries": parsed.max_retries,
            "generation_max_tokens": parsed.generation_max_tokens,
            "generation_temperature": parsed.generation_temperature,
            "judge_max_tokens": parsed.judge_max_tokens,
            "judge_temperature": parsed.judge_temperature,
        },
        "macro_metrics": macro_metrics,
        "books": {book_id: report.to_dict() for book_id, report in reports.items()},
    }

    write_report(parsed.output_path, payload)

    print("Macro metrics:")
    for key, value in sorted(macro_metrics.items()):
        print(f"  {key}: {value:.4f}")

    return macro_metrics


if __name__ == "__main__":  # pragma: no cover
    main()
