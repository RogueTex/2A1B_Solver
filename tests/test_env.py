"""Unit tests for the Alphabet 2048 Gym environment."""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest

from rl2048 import utils
from rl2048.env_alphabet2048 import (
    Alphabet2048Config,
    Alphabet2048Env,
    can_move,
    slide_and_merge,
)


@pytest.fixture
def env() -> Iterator[Alphabet2048Env]:
    config = Alphabet2048Config(seed=123, spawn_b_probability=0.0)
    instance = Alphabet2048Env(config)
    yield instance
    instance.close()


def test_reset_spawns_single_a(env: Alphabet2048Env) -> None:
    board, info = env.reset()
    assert board.shape == (4, 4)
    assert np.count_nonzero(board == 1) == 1
    assert info["largest_level"] == int(board.max())


def test_invalid_action_penalty_and_no_spawn(env: Alphabet2048Env) -> None:
    env.reset()
    env._board = np.zeros((4, 4), dtype=np.int32)
    env._board[0, 0] = 1  # Only tile already against the left wall
    before = env._board.copy()

    obs, reward, terminated, truncated, info = env.step(3)  # action=left

    assert np.array_equal(obs, before)
    assert pytest.approx(reward) == env.config.invalid_action_penalty
    assert not terminated and not truncated
    assert info["invalid_action"] is True


def test_merge_once_per_tile(env: Alphabet2048Env) -> None:
    env.reset()
    board = np.zeros((4, 4), dtype=np.int32)
    board[0, :3] = np.array([1, 1, 1])
    env._board = board

    before_spawn, _, _, _ = slide_and_merge(env._board, 3)
    obs, reward, *_ = env.step(3)

    expected = before_spawn.copy()
    diff = obs - expected
    spawn_locations = np.argwhere(diff != 0)
    assert reward >= utils.level_to_value(2)
    for row, col in spawn_locations:
        if expected[row, col] == 0 and obs[row, col] == 1:
            expected[row, col] = 1
    assert np.array_equal(obs, expected)


def test_terminal_when_no_moves(env: Alphabet2048Env) -> None:
    env.reset()
    env._board = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        dtype=np.int32,
    )
    obs, reward, terminated, _, info = env.step(0)
    assert terminated is True
    assert reward == 0.0
    assert can_move(obs) is False
    assert info["valid_actions"].sum() == 0


def test_mask_invalid_actions_matches_helper(env: Alphabet2048Env) -> None:
    env.reset()
    env._board = np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    mask = utils.mask_invalid_actions(env._board)
    assert mask.dtype == bool
    assert bool(mask[0]) is False  # up
    assert bool(mask[2]) is False  # down
    assert bool(mask[1]) is False  # right
    assert bool(mask[3]) is True   # left