"""Tests for curriculum, reward shaping, and diagnostics enhancements."""
from __future__ import annotations

import json

import numpy as np

from rl2048.curriculum import CurriculumConfig, CurriculumEnv
from rl2048.diagnostics import DecisionPoint, DiagnosticAnalyzer
from rl2048.env_alphabet2048 import Alphabet2048Config, Alphabet2048Env
from rl2048.reward_shaper import RewardShaper


def test_reward_shaper_is_finite_and_bounded() -> None:
    shaper = RewardShaper(
        merge_bonus_coef=0.05,
        entropy_penalty_coef=0.1,
        merge_preservation_coef=0.01,
        clip_abs=3.0,
    )
    before = np.array(
        [
            [1, 1, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    after = np.array(
        [
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    value = shaper.shape_reward(
        board_before=before,
        board_after=after,
        action=3,
        base_reward=2.0,
        done=False,
    )
    assert np.isfinite(value)
    assert -3.0 <= value <= 3.0


def test_env_accepts_reward_shaper() -> None:
    env = Alphabet2048Env(
        Alphabet2048Config(seed=123),
        reward_shaper=RewardShaper(clip_abs=5.0),
    )
    state, info = env.reset()
    valid = np.where(info["valid_actions"])[0]
    action = int(valid[0]) if len(valid) > 0 else 0
    _, reward, *_ = env.step(action)
    assert np.isfinite(reward)
    env.close()


def test_curriculum_env_density_bounds() -> None:
    env = CurriculumEnv(
        Alphabet2048Config(seed=7),
        CurriculumConfig(initial_board_density=2, max_board_density=5, schedule_type="linear"),
    )
    board, _ = env.reset(seed=7)
    filled = int(np.count_nonzero(board))
    assert 1 <= filled <= 15
    assert 2 <= env.curriculum.current_density() <= 5
    env.close()


def test_diagnostics_save_analysis(tmp_path) -> None:
    analyzer = DiagnosticAnalyzer()
    analyzer.log_decision(
        DecisionPoint(
            board_state=np.zeros((4, 4), dtype=np.int32),
            valid_actions=[0, 1, 2, 3],
            chosen_action=1,
            q_values=np.array([0.2, 0.8, 0.1, 0.0], dtype=np.float32),
            confidence=1.5,
            episode_return_so_far=10.0,
            steps_into_episode=4,
            game_outcome="won",
        )
    )
    output = tmp_path / "diagnostics.json"
    analyzer.save_analysis(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "failure_modes" in payload
    assert "regret_analysis" in payload
    assert payload["total_steps"] == 1
