"""Edge-case and integration tests for 2A1B Solver."""
from __future__ import annotations

import numpy as np
import pytest

from rl2048 import utils
from rl2048.curriculum import CurriculumConfig, CurriculumEnv
from rl2048.dqn_agent import DQNAgent, DQNConfig
from rl2048.env_alphabet2048 import (
    Alphabet2048Config,
    Alphabet2048Env,
    slide_and_merge,
)
from rl2048.reward_shaper import RewardShaper


# ---------------------------------------------------------------------------
# Environment edge cases
# ---------------------------------------------------------------------------


def test_reset_returns_correct_dtype() -> None:
      """Board dtype must be int32 after reset."""
      env = Alphabet2048Env(Alphabet2048Config(seed=0))
      board, _ = env.reset()
      assert board.dtype == np.int32
      env.close()


def test_step_observation_dtype() -> None:
      """Step must return int32 observations."""
      env = Alphabet2048Env(Alphabet2048Config(seed=1))
      obs, info = env.reset()
      valid = np.where(info["valid_actions"])[0]
      action = int(valid[0]) if len(valid) > 0 else 0
      obs2, *_ = env.step(action)
      assert obs2.dtype == np.int32
      env.close()


def test_multiple_resets_are_independent() -> None:
      """Two consecutive resets should produce fresh boards."""
      env = Alphabet2048Env(Alphabet2048Config(seed=42))
      board1, _ = env.reset()
      board2, _ = env.reset()
      # Boards may differ after second reset (new tile placement);
      # at minimum the environment should not raise and boards are valid.
      assert board1.shape == board2.shape == (4, 4)
      env.close()


def test_valid_actions_info_key() -> None:
      """Info dict must contain valid_actions as a boolean array."""
      env = Alphabet2048Env(Alphabet2048Config(seed=3))
      _, info = env.reset()
      assert "valid_actions" in info
      assert info["valid_actions"].dtype == bool
      assert info["valid_actions"].shape == (4,)
      env.close()


def test_slide_and_merge_all_same_row() -> None:
      """Sliding a row of identical tiles left should merge pairs."""
      board = np.array(
          [
              [1, 1, 1, 1],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
          ],
          dtype=np.int32,
      )
      result, _, reward, _ = slide_and_merge(board, action=3)  # left
    assert result[0, 0] == 2
    assert result[0, 1] == 2
    assert reward > 0


def test_slide_and_merge_no_merge_possible() -> None:
      """Sliding when no merges are possible should not change values."""
      board = np.array(
          [
              [1, 2, 3, 4],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
          ],
          dtype=np.int32,
      )
      result, changed, reward, _ = slide_and_merge(board, action=3)  # left
    assert reward == 0.0
    assert not changed  # nothing moved


# ---------------------------------------------------------------------------
# Agent edge cases
# ---------------------------------------------------------------------------


def _make_agent(**overrides) -> DQNAgent:
      config = DQNConfig(
                board_size=4,
                hidden_sizes=(32,),
                replay_capacity=200,
                batch_size=4,
                warmup_steps=0,
                epsilon_decay_steps=10,
                use_onehot=False,
                use_dueling=False,
                use_prioritized=False,
                **overrides,
      )
      return DQNAgent(config)


def test_epsilon_starts_at_one_and_decays() -> None:
      """Epsilon should start at 1.0 and decrease toward epsilon_min."""
      agent = _make_agent(epsilon_start=1.0, epsilon_min=0.1, epsilon_decay_steps=100)
      eps0 = agent.epsilon(train_step=0)
      eps50 = agent.epsilon(train_step=50)
      eps200 = agent.epsilon(train_step=200)
      assert eps0 == pytest.approx(1.0)
      assert eps50 < eps0
      assert eps200 == pytest.approx(0.1)  # clamped at min


def test_onehot_agent_forward_shape() -> None:
      """One-hot encoded agent should still produce (batch, 4) Q-values."""
      agent = _make_agent(use_onehot=True)
      states = np.zeros((3, 4, 4), dtype=np.int32)
      tensor = agent._states_to_tensor(states)
      q = agent.q_network(tensor)
      assert q.shape == (3, 4)


def test_dueling_agent_forward_no_nan() -> None:
      """Dueling DQN output must be finite."""
      agent = _make_agent(use_dueling=True)
      states = np.random.randint(0, 5, size=(4, 4, 4), dtype=np.int32)
      tensor = agent._states_to_tensor(states)
      q = agent.q_network(tensor).detach().cpu().numpy()
      assert not np.isnan(q).any()
      assert not np.isinf(q).any()


def test_select_action_all_invalid_falls_back() -> None:
      """When all actions are masked invalid, agent must still return an int."""
      agent = _make_agent()
      board = np.zeros((4, 4), dtype=np.int32)
      all_invalid = np.array([False, False, False, False])
      # Should not raise; behaviour may be arbitrary but type must be int.
      result = agent.select_action(board, all_invalid, train_step=0, explore=False)
    assert isinstance(result, (int, np.integer))


# ---------------------------------------------------------------------------
# Reward shaper edge cases
# ---------------------------------------------------------------------------


def test_reward_shaper_zero_coefs_returns_base() -> None:
      """With all shaping coefficients at 0 the output should equal base_reward."""
      shaper = RewardShaper(
          merge_bonus_coef=0.0,
          entropy_penalty_coef=0.0,
          merge_preservation_coef=0.0,
          clip_abs=10.0,
      )
      board = np.zeros((4, 4), dtype=np.int32)
      value = shaper.shape_reward(
          board_before=board,
          board_after=board,
          action=0,
          base_reward=3.5,
          done=False,
      )
      assert value == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Curriculum edge cases
# ---------------------------------------------------------------------------


def test_curriculum_density_never_exceeds_max() -> None:
      """Density reported by the curriculum must stay within configured bounds."""
      env = CurriculumEnv(
          Alphabet2048Config(seed=0),
          CurriculumConfig(initial_board_density=1, max_board_density=4, schedule_type="linear"),
      )
      for step in range(20):
                env.reset()
                density = env.curriculum.current_density()
                assert density <= 4, f"Density {density} exceeded max at step {step}"
            env.close()


def test_curriculum_adaptive_responds_to_wins() -> None:
      """Adaptive curriculum density should increase on successful episodes."""
    env = CurriculumEnv(
              Alphabet2048Config(seed=5),
              CurriculumConfig(
                            initial_board_density=1,
                            max_board_density=10,
                            schedule_type="adaptive",
              ),
    )
    initial_density = env.curriculum.current_density()
    # Simulate several successful episodes by stepping the curriculum.
    for _ in range(15):
              env.curriculum.step(won=True)
          assert env.curriculum.current_density() >= initial_density
    env.close()
