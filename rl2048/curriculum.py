"""Curriculum learning scheduler for progressive board complexity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .env_alphabet2048 import Alphabet2048Config, Alphabet2048Env


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning schedule."""

    initial_board_density: int = 2
    max_board_density: int = 12
    schedule_type: str = "linear"  # linear, exponential, adaptive
    steps_per_stage: int = 50_000
    adaptive_window: int = 20
    adaptive_target_return: float = 2_000.0
    adaptive_up_ratio: float = 1.2
    adaptive_down_ratio: float = 0.8


class CurriculumScheduler:
    """Track training progress and emit board-density targets."""

    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.total_steps = 0
        self.steps_in_stage = 0
        self.current_stage = 0
        self.recent_returns: list[float] = []
        self._adaptive_density_level = int(config.initial_board_density)

    def current_density(self) -> int:
        """Current board density (number of prefilled tiles)."""
        lo = max(1, int(self.config.initial_board_density))
        hi = max(lo, int(self.config.max_board_density))

        if self.config.schedule_type == "linear":
            horizon = max(self.config.steps_per_stage * 10, 1)
            progress = min(1.0, self.total_steps / horizon)
            return int(round(lo + (hi - lo) * progress))

        if self.config.schedule_type == "exponential":
            horizon = max(self.config.steps_per_stage * 20, 1)
            progress = min(1.0, self.total_steps / horizon)
            val = lo * ((hi / lo) ** progress)
            return int(round(val))

        # adaptive
        return int(np.clip(self._adaptive_density_level, lo, hi))

    def update(self, episode_return: float) -> None:
        """Update internal progress based on completed episode return."""
        self.recent_returns.append(float(episode_return))
        self.total_steps += 1
        self.steps_in_stage += 1

        if self.steps_in_stage >= self.config.steps_per_stage:
            self.current_stage += 1
            self.steps_in_stage = 0

        if self.config.schedule_type != "adaptive":
            return

        if len(self.recent_returns) < self.config.adaptive_window:
            return

        window = self.recent_returns[-self.config.adaptive_window :]
        avg_return = float(np.mean(window))
        target = self.config.adaptive_target_return

        if avg_return > target * self.config.adaptive_up_ratio:
            self._adaptive_density_level += 1
        elif avg_return < target * self.config.adaptive_down_ratio:
            self._adaptive_density_level -= 1

    def get_board_generator(self, board_size: int) -> Callable[[np.random.Generator], np.ndarray]:
        """Build a callable that generates boards at current difficulty."""
        density = int(self.current_density())
        max_tiles = board_size * board_size - 1  # keep at least one empty cell
        density = max(1, min(density, max_tiles))

        def generate_board(rng: np.random.Generator) -> np.ndarray:
            board = np.zeros((board_size, board_size), dtype=np.int32)
            positions = rng.choice(board_size * board_size, size=density, replace=False)
            for pos in positions:
                row, col = divmod(int(pos), board_size)
                board[row, col] = 2 if rng.random() < 0.1 else 1
            return board

        return generate_board


class CurriculumEnv(Alphabet2048Env):
    """Alphabet2048 environment wrapper with curriculum reset boards."""

    def __init__(
        self,
        base_env_config: Alphabet2048Config,
        curriculum_config: CurriculumConfig,
        *,
        reward_shaper: object | None = None,
    ) -> None:
        super().__init__(base_env_config, reward_shaper=reward_shaper)
        self.curriculum = CurriculumScheduler(curriculum_config)

    def reset(self, *, seed: int | None = None, options=None):  # type: ignore[override]
        obs, info = super().reset(seed=seed, options=options)
        board_gen = self.curriculum.get_board_generator(self.board_size)
        self._board = board_gen(self._rng)
        # Reset scores for a clean start from custom board.
        self._score = 0.0
        self._score_numeric = 0
        return self._board.copy(), self._info_dict(invalid_action=False)

    def step(self, action: int):  # type: ignore[override]
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            self.curriculum.update(float(info.get("score", 0.0)))
        return obs, reward, terminated, truncated, info
