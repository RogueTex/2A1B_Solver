"""Gymnasium environment for the Alphabet 2048 game."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from . import utils


@dataclass
class Alphabet2048Config:
    board_size: int = 4
    invalid_action_penalty: float = -1.0
    log_reward: bool = False
    seed: int | None = None
    spawn_b_probability: float = 0.1


def slide_and_merge(board: np.ndarray, action: int):
    """Helper that forwards to :func:`rl2048.utils.slide_and_merge`."""

    return utils.slide_and_merge(board, action)


def can_move(board: np.ndarray) -> bool:
    """Return ``True`` if any valid action remains."""

    return utils.can_move(board)


def spawn_new_tile(board: np.ndarray, rng: np.random.Generator, *, spawn_b_probability: float) -> bool:
    """Spawn an ``A`` tile (level 1) or, with a small probability, a ``B`` (level 2)."""

    empties = np.argwhere(board == 0)
    if empties.size == 0:
        return False
    choice = empties[rng.integers(len(empties))]
    level = 2 if rng.random() < spawn_b_probability else 1
    board[tuple(choice)] = level
    return True


class Alphabet2048Env(gym.Env[np.ndarray, int]):
    """Alphabet 2048 environment following the Gymnasium API."""

    metadata = {"render_modes": ["ansi"], "render_fps": 30}

    def __init__(
        self,
        config: Alphabet2048Config | None = None,
        *,
        reward_shaper: Optional[object] = None,
    ):
        super().__init__()
        self.config = config or Alphabet2048Config()
        self.reward_shaper = reward_shaper
        self.board_size = self.config.board_size
        self.observation_space = gym.spaces.Box(
            low=0,
            high=26,
            shape=(self.board_size, self.board_size),
            dtype=np.int32,
        )
        self.action_space = gym.spaces.Discrete(4)
        self._rng = np.random.default_rng(self.config.seed)
        self._board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self._score = 0.0
        self._score_numeric = 0

    # ------------------------------------------------------------------
    # Core Gym API
    # ------------------------------------------------------------------

    def seed(self, seed: int | None = None) -> None:
        if seed is not None:
            self.config.seed = seed
        self._rng = np.random.default_rng(self.config.seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:  # type: ignore[override]
        if seed is not None:
            self.seed(seed)
        self._board.fill(0)
        spawn_new_tile(
            self._board,
            self._rng,
            spawn_b_probability=self.config.spawn_b_probability,
        )
        self._score = 0.0
        self._score_numeric = 0
        info = self._info_dict(invalid_action=False)
        return self._board.copy(), info

    def step(self, action: int):  # type: ignore[override]
        if not self.action_space.contains(action):
            raise gym.error.InvalidAction(f"Invalid action {action}")

        if not can_move(self._board):
            info = self._info_dict(invalid_action=False)
            return self._board.copy(), 0.0, True, False, info

        board_before = self._board.copy()
        next_board, value_reward, moved, merged_levels = slide_and_merge(self._board, action)

        if not moved:
            reward = self.config.invalid_action_penalty
            self._score += reward
            info = self._info_dict(invalid_action=True)
            terminated = not can_move(self._board)
            return self._board.copy(), reward, terminated, False, info

        spawn_new_tile(
            next_board,
            self._rng,
            spawn_b_probability=self.config.spawn_b_probability,
        )

        self._board = next_board

        numeric_reward = float(value_reward)
        log_reward = float(sum(merged_levels))
        base_reward = log_reward if self.config.log_reward else numeric_reward
        terminated = not can_move(self._board)
        reward = base_reward
        if self.reward_shaper is not None:
            reward = float(
                self.reward_shaper.shape_reward(
                    board_before=board_before,
                    board_after=self._board.copy(),
                    action=action,
                    base_reward=base_reward,
                    done=terminated,
                )
            )

        self._score += reward
        self._score_numeric += int(numeric_reward)
        info = self._info_dict(invalid_action=False)
        return self._board.copy(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Rendering & helpers
    # ------------------------------------------------------------------

    def render(self):  # type: ignore[override]
        rows = []
        for row in self._board:
            rows.append("\t".join(utils.int_to_letter(int(val)) or "-" for val in row))
        return "\n".join(rows)

    def close(self):  # type: ignore[override]
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _info_dict(self, *, invalid_action: bool) -> Dict[str, Any]:
        valid_mask = (~utils.mask_invalid_actions(self._board)).copy()
        largest = int(self._board.max()) if self._board.size else 0
        return {
            "score": float(self._score),
            "score_numeric": int(self._score_numeric),
            "largest_level": largest,
            "valid_actions": valid_mask,
            "invalid_action": invalid_action,
        }
