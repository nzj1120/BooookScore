"""High level evaluation pipeline for AlignEval."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from .aligner import AlignmentConfig, align
from .encoder_bert import BertSentenceEncoder, EncoderConfig, normalize_embeddings
from .metrics import (
    MetricConfig,
    SentenceMetrics,
    compute_alignment_confidence,
    compute_coverage,
    compute_pfs,
    compute_scs,
)
from .report import BookReport
from .sim import compute_similarity_matrix, top_k_indices


@dataclass
class EvaluationConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    top_k: int = 3


def _sentence_payload(sentences: List[str], index: int) -> str:
    if 0 <= index < len(sentences):
        return sentences[index]
    return ""


def evaluate_alignment(
    book_id: str,
    source_text: str,
    summary_text: str,
    config: EvaluationConfig,
    encoder: Optional[BertSentenceEncoder] = None,
) -> Tuple[BookReport, Dict[str, float]]:
    own_encoder = encoder or BertSentenceEncoder(config.encoder)
    source_sentences = BertSentenceEncoder.split_sentences(source_text)
    summary_sentences = BertSentenceEncoder.split_sentences(summary_text)

    source_emb = normalize_embeddings(own_encoder.encode(source_sentences))
    summary_emb = normalize_embeddings(own_encoder.encode(summary_sentences))

    sim_matrix = compute_similarity_matrix(summary_emb, source_emb)
    if sim_matrix.size and sim_matrix.shape[1] > 0:
        best_indices = np.argmax(sim_matrix, axis=1)
        rows = np.arange(sim_matrix.shape[0])
        best_scores = sim_matrix[rows, best_indices]
    else:
        best_indices = np.full(sim_matrix.shape[0], -1, dtype=int)
        best_scores = np.zeros(sim_matrix.shape[0], dtype=float)

    if sim_matrix.shape[1] == 0:
        coverage, threshold = 0.0, 0.0
    else:
        coverage, threshold = compute_coverage(best_scores)
    alignment_pairs = align(sim_matrix, config.alignment)
    alignment_confidence = compute_alignment_confidence(alignment_pairs, sim_matrix)
    pfs, deviations = compute_pfs(
        alignment_pairs,
        sim_matrix,
        len(summary_sentences),
        len(source_sentences),
        config.metrics.pfs_gamma,
        config.metrics.epsilon,
    )

    top_indices, top_scores = top_k_indices(sim_matrix, config.top_k)
    scs, scs_scores = compute_scs(
        sim_matrix,
        top_indices,
        top_scores,
        config.metrics.scs_alpha,
        config.metrics.scs_beta,
    )

    sentence_entries: List[SentenceMetrics] = []
    aligned_map = {i: j for i, j in alignment_pairs}
    deviation_map: Dict[int, float] = {}
    for (i_idx, _), dev in zip(alignment_pairs, deviations):
        deviation_map[i_idx] = dev
    for idx, sent in enumerate(summary_sentences):
        best_idx = int(best_indices[idx]) if best_indices.size and best_indices[idx] >= 0 else None
        best_sim = float(best_scores[idx]) if best_scores.size else 0.0
        coverage_hit = bool(best_idx is not None and best_sim >= threshold and sim_matrix.shape[1] > 0)
        aligned_idx = aligned_map.get(idx)
        aligned_sim = float(sim_matrix[idx, aligned_idx]) if aligned_idx is not None else None
        position_dev = deviation_map.get(idx)
        scs_value = scs_scores[idx] if idx < len(scs_scores) else None
        topk_payload = []
        if top_indices.size:
            for j_idx, score in zip(top_indices[idx], top_scores[idx]):
                topk_payload.append(
                    {
                        "source_index": int(j_idx),
                        "similarity": float(score),
                        "source_sentence": _sentence_payload(source_sentences, int(j_idx)),
                    }
                )
        sentence_entries.append(
            SentenceMetrics(
                index=idx,
                sentence=sent,
                best_source_index=best_idx,
                best_similarity=best_sim,
                coverage_hit=coverage_hit,
                aligned_index=aligned_idx,
                aligned_similarity=aligned_sim,
                position_deviation=position_dev,
                scs=scs_value,
                topk=topk_payload,
            )
        )

    global_metrics = {
        "coverage": coverage,
        "coverage_threshold": threshold,
        "alignment_confidence": alignment_confidence,
        "pfs": pfs,
        "scs": scs,
        "summary_sentence_count": float(len(summary_sentences)),
        "source_sentence_count": float(len(source_sentences)),
    }
    report = BookReport(
        book_id=book_id,
        global_metrics=global_metrics,
        sentences=sentence_entries,
        alignment_path=[(int(i), int(j)) for i, j in alignment_pairs],
    )
    return report, global_metrics


def _progress_iter(
    items: Iterable[Tuple[str, str]],
    total: int,
    enabled: bool,
) -> Iterator[Tuple[str, str]]:
    if not enabled:
        yield from items
        return
    yield from tqdm(items, total=total, desc="Evaluating summaries")


def evaluate_corpus(
    source_map: Dict[str, str],
    summary_map: Dict[str, str],
    config: EvaluationConfig,
    show_progress: bool = False,
) -> Tuple[Dict[str, BookReport], Dict[str, float]]:
    encoder = BertSentenceEncoder(config.encoder)
    reports: Dict[str, BookReport] = {}
    metrics_total: Dict[str, float] = {}
    count = 0
    valid_items = [(bid, summary_map[bid]) for bid in summary_map if bid in source_map]
    for book_id, summary_text in _progress_iter(valid_items, len(valid_items), show_progress):
        if book_id not in source_map:
            continue
        source_text = source_map[book_id]
        report, metrics = evaluate_alignment(book_id, source_text, summary_text, config, encoder=encoder)
        reports[book_id] = report
        for key, value in metrics.items():
            metrics_total[key] = metrics_total.get(key, 0.0) + float(value)
        count += 1
    macro = {key: value / count for key, value in metrics_total.items()} if count else {}
    return reports, macro
