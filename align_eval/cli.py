"""Command line interface for AlignEval."""
from __future__ import annotations

import argparse
import json
from typing import Dict

from tqdm.auto import tqdm

from .aligner import AlignmentConfig
from .encoder_bert import EncoderConfig, BertSentenceEncoder
from .eval import EvaluationConfig, evaluate_alignment
from .metrics import MetricConfig
from .report import BookReport, write_report


def _load_json(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at {path}")
    return {str(k): str(v) for k, v in data.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate summary faithfulness via sentence alignment")
    parser.add_argument("--source_path", required=True, help="Path to original text JSON")
    parser.add_argument("--summary_path", required=True, help="Path to summary JSON")
    parser.add_argument("--output_path", required=True, help="Where to save evaluation report JSON")

    parser.add_argument("--model_name", default="hfl/chinese-bert-wwm-ext", help="Chinese BERT model name or path")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean", help="Pooling strategy for sentence embeddings")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default=None, help="Torch device for encoding")

    parser.add_argument("--alignment", choices=["nw", "dtw"], default="nw", help="Alignment algorithm")
    parser.add_argument("--gap_penalty", type=float, default=-0.5)
    parser.add_argument("--bandwidth", type=int, default=None, help="Optional alignment bandwidth")

    parser.add_argument("--pfs_gamma", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=10.0, help="Softmax temperature for SCS")
    parser.add_argument("--scs_beta", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=3)

    return parser


def main(args: argparse.Namespace | None = None) -> Dict[str, float]:
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    source_data = _load_json(parsed.source_path)
    summary_data = _load_json(parsed.summary_path)

    missing = sorted(set(summary_data) - set(source_data))
    if missing:
        raise ValueError(f"Missing source text for: {missing}")

    encoder_cfg = EncoderConfig(
        model_name=parsed.model_name,
        pooling=parsed.pooling,
        max_length=parsed.max_length,
        batch_size=parsed.batch_size,
        device=parsed.device,
    )
    alignment_cfg = AlignmentConfig(
        algorithm=parsed.alignment,
        gap_penalty=parsed.gap_penalty,
        bandwidth=parsed.bandwidth,
    )
    metric_cfg = MetricConfig(
        pfs_gamma=parsed.pfs_gamma,
        scs_alpha=parsed.alpha,
        scs_beta=parsed.scs_beta,
    )
    eval_cfg = EvaluationConfig(
        encoder=encoder_cfg,
        alignment=alignment_cfg,
        metrics=metric_cfg,
        top_k=parsed.top_k,
    )

    encoder = BertSentenceEncoder(encoder_cfg)

    book_reports: Dict[str, BookReport] = {}
    macro_metrics: Dict[str, float] = {}
    metric_accumulator: Dict[str, float] = {}
    count = 0

    iterator = summary_data.items()
    iterator = tqdm(iterator, total=len(summary_data), desc="Evaluating summaries")
    for book_id, summary_text in iterator:
        source_text = source_data[book_id]
        report, metrics = evaluate_alignment(book_id, source_text, summary_text, eval_cfg, encoder=encoder)
        book_reports[book_id] = report
        for key, value in metrics.items():
            metric_accumulator[key] = metric_accumulator.get(key, 0.0) + float(value)
        count += 1

    if count:
        macro_metrics = {key: value / count for key, value in metric_accumulator.items()}

    summary_payload = {
        "config": {
            "model_name": parsed.model_name,
            "pooling": parsed.pooling,
            "alignment": parsed.alignment,
            "gap_penalty": parsed.gap_penalty,
            "bandwidth": parsed.bandwidth,
            "pfs_gamma": parsed.pfs_gamma,
            "alpha": parsed.alpha,
            "scs_beta": parsed.scs_beta,
            "top_k": parsed.top_k,
        },
        "macro_metrics": macro_metrics,
        "books": {book_id: report.to_dict() for book_id, report in book_reports.items()},
    }

    write_report(parsed.output_path, summary_payload)

    if macro_metrics:
        print("Macro metrics:")
        for key, value in sorted(macro_metrics.items()):
            print(f"  {key}: {value:.4f}")
    else:
        print("No books evaluated.")

    return macro_metrics


if __name__ == "__main__":
    main()
