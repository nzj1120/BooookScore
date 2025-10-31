"""Monotonic alignment utilities (Needleman–Wunsch / DTW variants)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class AlignmentConfig:
    algorithm: str = "nw"
    gap_penalty: float = -0.5
    bandwidth: int | None = None


def _apply_band_constraints(i: int, n_summary: int, n_source: int, bandwidth: int | None) -> Tuple[int, int]:
    if bandwidth is None:
        return 1, n_source
    # bandwidth interpreted in absolute sentence count
    start = max(1, i - bandwidth)
    end = min(n_source, i + bandwidth)
    return start, end


def needleman_wunsch(sim_matrix: np.ndarray, gap_penalty: float, bandwidth: int | None) -> List[Tuple[int, int]]:
    m, n = sim_matrix.shape
    dp = np.full((m + 1, n + 1), -np.inf, dtype=np.float32)
    trace = np.zeros((m + 1, n + 1), dtype=np.int8)  # 0=diag,1=up,2=left
    dp[0, 0] = 0.0
    for i in range(1, m + 1):
        dp[i, 0] = gap_penalty * i
        trace[i, 0] = 1
    for j in range(1, n + 1):
        dp[0, j] = gap_penalty * j
        trace[0, j] = 2
    for i in range(1, m + 1):
        j_start, j_end = _apply_band_constraints(i, m, n, bandwidth)
        for j in range(j_start, j_end + 1):
            match = dp[i - 1, j - 1] + sim_matrix[i - 1, j - 1]
            delete = dp[i - 1, j] + gap_penalty
            insert = dp[i, j - 1] + gap_penalty
            candidates = [match, delete, insert]
            best = max(candidates)
            dp[i, j] = best
            trace[i, j] = int(candidates.index(best))
        # invalidate cells outside the band to keep monotonicity constraints
        if bandwidth is not None:
            for j in range(1, j_start):
                dp[i, j] = -np.inf
            for j in range(j_end + 1, n + 1):
                dp[i, j] = -np.inf
    i, j = m, n
    if np.isneginf(dp[i, j]):
        # fallback：选择最佳终点，避免带宽导致无路径
        idx = np.argmax(dp)
        i, j = np.unravel_index(idx, dp.shape)
    path: List[Tuple[int, int]] = []
    while i > 0 or j > 0:
        move = trace[i, j]
        if move == 0:
            i -= 1
            j -= 1
            path.append((i, j))
        elif move == 1:
            i -= 1
        else:
            j -= 1
        if i == 0 and j == 0:
            break
    path.reverse()
    return path


def dynamic_time_warping(sim_matrix: np.ndarray, bandwidth: int | None) -> List[Tuple[int, int]]:
    # Implement DTW maximizing similarity by minimizing negative similarity cost.
    m, n = sim_matrix.shape
    cost = np.full((m + 1, n + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0
    trace = np.zeros((m + 1, n + 1), dtype=np.int8)
    for i in range(1, m + 1):
        j_start, j_end = _apply_band_constraints(i, m, n, bandwidth)
        for j in range(j_start, j_end + 1):
            sim = sim_matrix[i - 1, j - 1]
            penalties = [cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]]
            idx = int(np.argmin(penalties))
            cost[i, j] = penalties[idx] + (1.0 - sim)
            trace[i, j] = idx
    i, j = m, n
    if np.isinf(cost[i, j]):
        idx = np.argmin(cost)
        i, j = np.unravel_index(idx, cost.shape)
    path: List[Tuple[int, int]] = []
    while i > 0 and j > 0:
        move = trace[i, j]
        if move == 2:  # diag
            i -= 1
            j -= 1
            path.append((i, j))
        elif move == 0:  # up
            i -= 1
        else:  # left
            j -= 1
        if i == 0 and j == 0:
            path.append((0, 0))
            break
    path = [(i, j) for i, j in path if i < m and j < n]
    path.sort()
    return path


def align(sim_matrix: np.ndarray, config: AlignmentConfig) -> List[Tuple[int, int]]:
    algorithm = config.algorithm.lower()
    if algorithm == "nw":
        return needleman_wunsch(sim_matrix, config.gap_penalty, config.bandwidth)
    if algorithm == "dtw":
        return dynamic_time_warping(sim_matrix, config.bandwidth)
    raise ValueError(f"Unsupported alignment algorithm: {config.algorithm}")
