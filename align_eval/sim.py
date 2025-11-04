"""Similarity utilities for AlignEval."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_similarity_matrix(summary_emb: np.ndarray, source_emb: np.ndarray) -> np.ndarray:
    """Computes cosine similarity matrix between summary and source embeddings."""
    if summary_emb.size == 0 or source_emb.size == 0:
        return np.zeros((summary_emb.shape[0], source_emb.shape[0]), dtype=np.float32)
    sim = np.matmul(summary_emb, source_emb.T)
    return sim.astype(np.float32)


def top_k_indices(sim_matrix: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns top-k indices and scores per summary sentence."""
    if sim_matrix.size == 0 or sim_matrix.shape[1] == 0 or k <= 0:
        return (
            np.zeros((sim_matrix.shape[0], 0), dtype=np.int32),
            np.zeros((sim_matrix.shape[0], 0), dtype=np.float32),
        )
    k = min(k, sim_matrix.shape[1])
    partition_indices = np.argpartition(-sim_matrix, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(sim_matrix.shape[0])[:, None]
    scores = sim_matrix[rows, partition_indices]
    order = np.argsort(-scores, axis=1)
    sorted_indices = partition_indices[rows, order]
    sorted_scores = scores[rows, order]
    return sorted_indices.astype(np.int32), sorted_scores.astype(np.float32)
