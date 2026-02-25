"""Evaluate a trained DQN policy on Alphabet 2048."""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from collections import Counter
from typing import Dict, List

import numpy as np
import torch

from rl2048 import utils
from rl2048.diagnostics import DecisionPoint, DiagnosticAnalyzer
from rl2048.dqn_agent import DQNAgent, DQNConfig
from rl2048.env_alphabet2048 import Alphabet2048Config, Alphabet2048Env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Model checkpoint path")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for evaluation episodes")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for loading the model")
    parser.add_argument("--log-reward", action="store_true", help="Evaluate using log reward shaping")
    parser.add_argument("--target-letter", type=str, default="K", help="Report success rate for reaching this tile")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional JSON file to store per-episode summaries",
    )
    parser.add_argument("--save-diagnostics", action="store_true", help="Save diagnostics JSON report")
    parser.add_argument(
        "--diagnostics-output",
        type=pathlib.Path,
        default=pathlib.Path("diagnostics.json"),
        help="Diagnostics output path when --save-diagnostics is set",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    utils.seed_everything(args.seed)

    env_config = Alphabet2048Config(
        seed=args.seed,
        log_reward=args.log_reward,
    )

    agent = DQNAgent(DQNConfig(device=args.device))
    agent.load(str(args.checkpoint))

    per_episode: List[Dict[str, float | int]] = []
    rewards: List[float] = []
    lengths: List[int] = []
    largest_levels: List[int] = []
    invalid_counts: List[int] = []

    target_level = utils.letter_to_int(args.target_letter)
    success_count = 0
    analyzer = DiagnosticAnalyzer() if args.save_diagnostics else None

    for idx in range(args.episodes):
        env = Alphabet2048Env(env_config)
        state, info = env.reset(seed=args.seed + idx)
        valid_mask = info["valid_actions"].astype(bool)
        done = False
        episode_reward = 0.0
        steps = 0
        invalid_moves = 0
        episode_start = len(analyzer.decision_log) if analyzer is not None else 0

        while not done:
            if analyzer is not None:
                state_tensor = agent._states_to_tensor(np.expand_dims(state, 0))
                with torch.no_grad():
                    q_values = agent.q_network(state_tensor).cpu().numpy()[0]
                valid_actions = np.where(valid_mask)[0]
                valid_q = q_values[valid_actions] if valid_actions.size > 0 else q_values
                max_valid_q = float(np.max(valid_q)) if valid_q.size > 0 else float(np.max(q_values))
                mean_valid_q = float(np.mean(valid_q)) if valid_q.size > 0 else 0.0
                confidence = max_valid_q / (abs(mean_valid_q) + 1e-6)

            action = agent.select_action(
                state,
                valid_mask,
                train_step=agent.config.epsilon_decay_steps,
                explore=False,
            )
            if analyzer is not None:
                analyzer.log_decision(
                    DecisionPoint(
                        board_state=state.copy(),
                        valid_actions=np.where(valid_mask)[0].tolist(),
                        chosen_action=int(action),
                        q_values=q_values.copy(),
                        confidence=float(confidence),
                        episode_return_so_far=float(episode_reward),
                        steps_into_episode=int(steps),
                    )
                )
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            if info.get("invalid_action"):
                invalid_moves += 1
            state = next_state
            valid_mask = info.get("valid_actions", np.ones(utils.NUM_ACTIONS, dtype=bool)).astype(bool)

        largest_level = int(info.get("largest_level", 0))
        if analyzer is not None:
            outcome = "won" if largest_level >= target_level else "lost"
            analyzer.set_episode_outcome(episode_start, outcome)
        success_count += int(largest_level >= target_level)
        rewards.append(episode_reward)
        lengths.append(steps)
        largest_levels.append(largest_level)
        invalid_counts.append(invalid_moves)
        per_episode.append(
            {
                "episode": idx,
                "return": episode_reward,
                "length": steps,
                "largest_level": largest_level,
                "invalid_moves": invalid_moves,
            }
        )
        env.close()

    mean_return = float(statistics.mean(rewards)) if rewards else 0.0
    median_return = float(statistics.median(rewards)) if rewards else 0.0
    lengths_mean = float(statistics.mean(lengths)) if lengths else 0.0
    invalid_rate = float(statistics.mean(i / max(l, 1) for i, l in zip(invalid_counts, lengths))) if lengths else 0.0
    dist = Counter(utils.int_to_letter(level) or "-" for level in largest_levels)
    success_rate = success_count / max(len(rewards), 1)

    print("=== Evaluation Summary ===")
    print(f"Episodes: {len(rewards)}")
    print(f"Mean return: {mean_return:.2f}")
    print(f"Median return: {median_return:.2f}")
    print(f"Mean length: {lengths_mean:.2f}")
    print(f"Invalid action rate: {invalid_rate:.2%}")
    print(f"Success (>= {args.target_letter}): {success_rate:.2%}")
    print("Largest tile distribution:")
    for letter, count in dist.most_common():
        print(f"  {letter or '-'}: {count}")

    summary = {
        "mean_return": mean_return,
        "median_return": median_return,
        "mean_length": lengths_mean,
        "invalid_rate": invalid_rate,
        "success_rate": success_rate,
        "largest_distribution": dict(dist),
    }

    output_path = args.output or args.checkpoint.with_suffix(".eval.json")
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump({"summary": summary, "episodes": per_episode}, fp, indent=2)
    print(f"Wrote evaluation details to {output_path}")

    if analyzer is not None:
        analyzer.save_analysis(args.diagnostics_output)
        print(f"Wrote diagnostics to {args.diagnostics_output}")


if __name__ == "__main__":
    main()
