"""Unit tests for the DQN agent."""
from __future__ import annotations

import numpy as np

from rl2048.dqn_agent import DQNAgent, DQNConfig


def make_agent(**overrides) -> DQNAgent:
    config = DQNConfig(
        board_size=4,
        hidden_sizes=(32,),
        replay_capacity=100,
        batch_size=2,
        warmup_steps=0,
        epsilon_decay_steps=1,
        use_onehot=False,
        use_dueling=False,
        use_prioritized=False,
        **overrides,
    )
    return DQNAgent(config)


def test_qnetwork_forward_shape_and_nan() -> None:
    agent = make_agent()
    states = np.zeros((2, 4, 4), dtype=np.int32)
    tensor = agent._states_to_tensor(states)
    q_values = agent.q_network(tensor)
    assert q_values.shape == (2, 4)
    assert not np.isnan(q_values.detach().cpu().numpy()).any()


def test_select_action_respects_mask() -> None:
    agent = make_agent()
    board = np.zeros((4, 4), dtype=np.int32)
    valid_mask = np.array([False, True, False, False])

    action = agent.select_action(board, valid_mask, train_step=0, explore=False)
    assert action == 1

    # Even with exploration enabled, only valid actions should be sampled
    valid_mask = np.array([True, False, True, False])
    for _ in range(100):
        sampled = agent.select_action(board, valid_mask, train_step=0, explore=True)
        assert sampled in {0, 2}


def test_update_reduces_td_error() -> None:
    agent = make_agent()
    state = np.zeros((4, 4), dtype=np.int32)
    next_state = state.copy()
    reward = 5.0
    done = True
    valid_mask = np.ones(4, dtype=bool)

    state_tensor = agent._states_to_tensor(np.expand_dims(state, 0))
    initial_q = agent.q_network(state_tensor)[0, 0].item()
    initial_td_error = abs(reward - initial_q)

    transition = (state, 0, reward, next_state, done, valid_mask)
    agent.update([transition, transition])

    updated_q = agent.q_network(state_tensor)[0, 0].item()
    updated_td_error = abs(reward - updated_q)

    assert updated_td_error < initial_td_error