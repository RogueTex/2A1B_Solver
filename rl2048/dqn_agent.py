"""Deep Q-Network agent with optional prioritized replay and dueling heads."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from . import utils


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]


@dataclass
class DQNConfig:
    board_size: int = 4
    max_level: int = 16
    hidden_sizes: Sequence[int] = (512, 512)
    replay_capacity: int = 200_000
    batch_size: int = 1_024
    warmup_steps: int = 20_000
    gamma: float = 0.99
    learning_rate: float = 1e-4
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 500_000
    gradient_clip: float = 1.0
    use_onehot: bool = True
    use_dueling: bool = True
    use_prioritized: bool = False
    prioritized_alpha: float = 0.6
    prioritized_beta_start: float = 0.4
    prioritized_beta_end: float = 1.0
    prioritized_beta_steps: int = 500_000
    prioritized_epsilon: float = 1e-6
    device: str = "cpu"


class QNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        *,
        action_dim: int,
        use_dueling: bool,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            prev = hidden
        self.feature = nn.Sequential(*layers) if layers else nn.Identity()
        last_dim = prev
        self.use_dueling = use_dueling
        if use_dueling:
            self.value_head = nn.Sequential(
                nn.Linear(last_dim, last_dim),
                nn.ReLU(),
                nn.Linear(last_dim, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(last_dim, last_dim),
                nn.ReLU(),
                nn.Linear(last_dim, action_dim),
            )
        else:
            self.q_head = nn.Linear(last_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        features = self.feature(x)
        if self.use_dueling:
            advantages = self.advantage_head(features)
            values = self.value_head(features)
            q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
        else:
            q_values = self.q_head(features)
        return q_values


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        *,
        prioritized: bool,
        alpha: float,
        epsilon: float,
    ) -> None:
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.position = 0
        self.prioritized = prioritized
        self.alpha = alpha
        self.epsilon = epsilon
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: Transition) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        priority = self.priorities.max() if self.buffer else 1.0
        if priority <= 0:
            priority = 1.0
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        *,
        beta: float,
    ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        if not self.buffer:
            raise RuntimeError("Replay buffer is empty")

        indices: np.ndarray
        weights = np.ones(batch_size, dtype=np.float32)
        if self.prioritized:
            current_priorities = self.priorities[: len(self.buffer)]
            scaled = np.power(current_priorities + self.epsilon, self.alpha)
            probs = scaled / scaled.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            weights = np.power(len(self.buffer) * probs[indices], -beta)
            weights = weights / weights.max()
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        if not self.prioritized:
            return
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority + self.epsilon)


class DQNAgent:
    def __init__(self, config: DQNConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        input_dim = (
            (config.max_level + 1) * config.board_size * config.board_size
            if config.use_onehot
            else config.board_size * config.board_size
        )
        self.q_network = QNetwork(
            input_dim,
            config.hidden_sizes,
            action_dim=utils.NUM_ACTIONS,
            use_dueling=config.use_dueling,
        ).to(self.device)
        self.target_network = QNetwork(
            input_dim,
            config.hidden_sizes,
            action_dim=utils.NUM_ACTIONS,
            use_dueling=config.use_dueling,
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(
            config.replay_capacity,
            prioritized=config.use_prioritized,
            alpha=config.prioritized_alpha,
            epsilon=config.prioritized_epsilon,
        )
        self.total_env_steps = 0
        self.total_updates = 0

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def select_action(
        self,
        state: np.ndarray,
        valid_mask: Optional[np.ndarray],
        train_step: int,
        *,
        explore: bool = True,
    ) -> int:
        epsilon = self._epsilon(train_step) if explore else 0.0
        self.total_env_steps = max(self.total_env_steps, train_step)
        if valid_mask is None:
            valid_mask = np.ones(utils.NUM_ACTIONS, dtype=bool)

        valid_indices = np.where(valid_mask)[0]
        if valid_indices.size == 0:
            valid_indices = np.arange(utils.NUM_ACTIONS)

        if np.random.rand() < epsilon:
            return int(np.random.choice(valid_indices))

        state_tensor = self._states_to_tensor(np.expand_dims(state, 0))
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        masked_q = np.full_like(q_values, fill_value=-1e9, dtype=np.float32)
        masked_q[valid_indices] = q_values[valid_indices]
        return int(np.argmax(masked_q))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, transitions: Iterable[Transition] | None = None) -> Optional[dict]:
        if transitions is not None:
            for transition in transitions:
                self.replay.push(transition)

        if (
            len(self.replay) < self.config.batch_size
            or self.total_env_steps < self.config.warmup_steps
        ):
            return None

        beta = self._beta(self.total_updates)
        batch, indices, weights = self.replay.sample(
            self.config.batch_size, beta=beta
        )
        states, actions, rewards, next_states, dones, next_valid = zip(*batch)

        states_tensor = self._states_to_tensor(np.stack(states))
        next_states_tensor = self._states_to_tensor(np.stack(next_states))
        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights_tensor = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_valid_tensor = torch.as_tensor(np.stack(next_valid), dtype=torch.bool, device=self.device)

        q_values = self.q_network(states_tensor).gather(1, actions_tensor)

        with torch.no_grad():
            next_q_online = self.q_network(next_states_tensor)
            next_q_online[~next_valid_tensor] = -1e9
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)

            next_q_target = self.target_network(next_states_tensor)
            next_q_target[~next_valid_tensor] = -1e9
            target_q = next_q_target.gather(1, next_actions)
            targets = rewards_tensor + self.config.gamma * (1 - dones_tensor) * target_q

        td_errors = targets - q_values
        loss = (weights_tensor * nn.functional.smooth_l1_loss(q_values, targets, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        self._soft_update()
        self.total_updates += 1

        if self.config.use_prioritized:
            priorities = torch.abs(td_errors).detach().cpu().numpy().flatten()
            self.replay.update_priorities(indices, priorities)

        return {
            "loss": float(loss.item()),
            "epsilon": float(self._epsilon(self.total_env_steps)),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        payload = {
            "config": asdict(self.config),
            "state_dict": self.q_network.state_dict(),
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        state_dict = payload["state_dict"] if "state_dict" in payload else payload
        self.q_network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _soft_update(self) -> None:
        tau = self.config.tau
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def _states_to_tensor(self, states: np.ndarray) -> torch.Tensor:
        if self.config.use_onehot:
            encodings = [utils.encode_onehot(state, self.config.max_level).reshape(-1) for state in states]
            feat = np.stack(encodings).astype(np.float32)
        else:
            values = utils.level_to_value(states).astype(np.float32)
            feat = np.log2(values + 1.0) / (self.config.max_level + 1)
            feat = feat.reshape(states.shape[0], -1)
        return torch.from_numpy(feat).to(self.device)

    def _epsilon(self, train_step: int) -> float:
        frac = min(1.0, max(0.0, train_step / max(1, self.config.epsilon_decay_steps)))
        return float(
            self.config.epsilon_end
            + (self.config.epsilon_start - self.config.epsilon_end) * (1.0 - frac)
        )

    def _beta(self, update_step: int) -> float:
        if not self.config.use_prioritized:
            return 1.0
        frac = min(1.0, update_step / max(1, self.config.prioritized_beta_steps))
        return float(
            self.config.prioritized_beta_start
            + frac * (self.config.prioritized_beta_end - self.config.prioritized_beta_start)
        )
