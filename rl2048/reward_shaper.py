"""Auxiliary reward signals for improving training convergence."""
from __future__ import annotations

import numpy as np


class RewardShaper:
    """Compute bounded auxiliary rewards based on board structure."""

    def __init__(
        self,
        *,
        use_merge_bonus: bool = True,
        use_entropy_penalty: bool = True,
        use_merge_preservation: bool = True,
        merge_bonus_coef: float = 0.05,
        entropy_penalty_coef: float = 0.1,
        merge_preservation_coef: float = 0.01,
        clip_abs: float = 10.0,
    ) -> None:
        self.use_merge_bonus = bool(use_merge_bonus)
        self.use_entropy_penalty = bool(use_entropy_penalty)
        self.use_merge_preservation = bool(use_merge_preservation)
        self.merge_bonus_coef = float(merge_bonus_coef)
        self.entropy_penalty_coef = float(entropy_penalty_coef)
        self.merge_preservation_coef = float(merge_preservation_coef)
        self.clip_abs = float(max(0.1, clip_abs))

    def shape_reward(
        self,
        *,
        board_before: np.ndarray,
        board_after: np.ndarray,
        action: int,
        base_reward: float,
        done: bool,
    ) -> float:
        """Augment base reward with bounded heuristic signals."""
        del action, board_before  # reserved for future shaping terms
        shaped = float(base_reward)

        if not done:
            if self.use_merge_preservation:
                shaped += self.merge_preservation_coef * self._count_adjacent_same(board_after)
            if self.use_entropy_penalty:
                shaped -= self.entropy_penalty_coef * self._board_entropy(board_after)
            if self.use_merge_bonus:
                shaped += self.merge_bonus_coef * self._count_merge_chains(board_after)

        return float(np.clip(shaped, -self.clip_abs, self.clip_abs))

    def _count_adjacent_same(self, board: np.ndarray) -> float:
        count = 0
        rows, cols = board.shape
        for i in range(rows):
            for j in range(cols - 1):
                if board[i, j] > 0 and board[i, j] == board[i, j + 1]:
                    count += 1
        for i in range(rows - 1):
            for j in range(cols):
                if board[i, j] > 0 and board[i, j] == board[i + 1, j]:
                    count += 1
        return float(count)

    def _board_entropy(self, board: np.ndarray) -> float:
        rows, cols = board.shape
        half_r = rows // 2
        half_c = cols // 2
        quadrants = np.array(
            [
                np.count_nonzero(board[:half_r, :half_c]),
                np.count_nonzero(board[:half_r, half_c:]),
                np.count_nonzero(board[half_r:, :half_c]),
                np.count_nonzero(board[half_r:, half_c:]),
            ],
            dtype=np.float32,
        )
        total = float(quadrants.sum())
        if total <= 0:
            return 0.0
        probs = quadrants / total
        nz = probs[probs > 0]
        return float(-(nz * np.log2(nz)).sum())

    def _count_merge_chains(self, board: np.ndarray) -> float:
        chains = 0
        for row in board:
            chains += self._line_chain_count(row)
        for col in board.T:
            chains += self._line_chain_count(col)
        return float(chains)

    @staticmethod
    def _line_chain_count(line: np.ndarray) -> int:
        chain_len = 1
        score = 0
        for i in range(1, len(line)):
            if line[i] > 0 and line[i] == line[i - 1]:
                chain_len += 1
            else:
                if chain_len >= 3:
                    score += chain_len - 2
                chain_len = 1
        if chain_len >= 3:
            score += chain_len - 2
        return score
