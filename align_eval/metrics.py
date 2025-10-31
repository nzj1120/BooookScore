"""Faithfulness metrics for AlignEval."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class MetricConfig:
    pfs_gamma: float = 3.0
    scs_alpha: float = 10.0
    scs_beta: float = 0.1
    epsilon: float = 1e-6


@dataclass
class SentenceMetrics:
    index: int
    sentence: str
    best_source_index: int | None
    best_similarity: float
    coverage_hit: bool
    aligned_index: int | None
    aligned_similarity: float | None
    position_deviation: float | None
    scs: float | None
    topk: List[Dict[str, float]]


def compute_coverage(best_sims: np.ndarray) -> Tuple[float, float]:
    if best_sims.size == 0:
        return 0.0, 0.0
    threshold = float(best_sims.mean())
    coverage = float((best_sims >= threshold).sum() / best_sims.shape[0])
    return coverage, threshold


def compute_alignment_confidence(pairs: Sequence[Tuple[int, int]], sim_matrix: np.ndarray) -> float:
    if not pairs:
        return 0.0
    sims = [float(sim_matrix[i, j]) for i, j in pairs]
    return float(np.mean(sims))


def compute_pfs(
    pairs: Sequence[Tuple[int, int]],
    sim_matrix: np.ndarray,
    m_summary: int,
    n_source: int,
    gamma: float,
    epsilon: float,
) -> Tuple[float, List[float]]:
    if not pairs or m_summary == 0 or n_source == 0:
        return 0.0, []
    deviations: List[float] = []
    weights: List[float] = []
    for i, j in pairs:
        u = (i + 0.5) / max(m_summary, 1)
        v = (j + 0.5) / max(n_source, 1)
        d = abs(u - v)
        deviations.append(d)
        weights.append(float(max(sim_matrix[i, j], epsilon)))
    numerator = float(np.dot(weights, deviations))
    denominator = float(np.sum(weights))
    avg_dev = numerator / max(denominator, epsilon)
    pfs = float(np.power(max(0.0, 1.0 - avg_dev), gamma))
    return pfs, deviations


def compute_scs(
    sim_matrix: np.ndarray,
    top_indices: np.ndarray,
    top_scores: np.ndarray,
    alpha: float,
    beta: float,
) -> Tuple[float, List[float]]:
    if sim_matrix.shape[0] == 0:
        return 0.0, []
    n_source = max(sim_matrix.shape[1], 1)
    scs_scores: List[float] = []
    for row in range(sim_matrix.shape[0]):
        if top_indices.shape[1] == 0:
            scs_scores.append(0.0)
            continue
        indices = top_indices[row]
        scores = top_scores[row]
        positions = (indices + 0.5) / n_source
        if np.all(scores == -np.inf):
            scs_scores.append(0.0)
            continue
        weights = np.exp(alpha * (scores - scores.max()))
        weights_sum = weights.sum()
        if weights_sum == 0:
            scs_scores.append(0.0)
            continue
        probs = weights / weights_sum
        mean_pos = float(np.sum(probs * positions))
        variance = float(np.sum(probs * (positions - mean_pos) ** 2))
        risk = min(max(variance / beta, 0.0), 1.0)
        scs_scores.append(1.0 - risk)
    return float(np.mean(scs_scores)), scs_scores
