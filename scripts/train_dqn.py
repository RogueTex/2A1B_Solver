"""Command-line training loop for the Alphabet 2048 DQN agent."""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
from collections import Counter, deque
from typing import Dict, List

import numpy as np

from rl2048 import utils
from rl2048.curriculum import CurriculumConfig, CurriculumEnv
from rl2048.dqn_agent import DQNAgent, DQNConfig, Transition
from rl2048.env_alphabet2048 import Alphabet2048Config, Alphabet2048Env
from rl2048.reward_shaper import RewardShaper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent on Alphabet 2048")
    parser.add_argument("--total-steps", type=int, default=1_000_000, help="Total training environment steps")
    parser.add_argument("--target-avg-return", type=float, default=4_000.0, help="Early-stop once eval mean surpasses this")
    parser.add_argument(
        "--checkpoint-dir",
        type=pathlib.Path,
        default=pathlib.Path("checkpoints"),
        help="Directory to store model checkpoints",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--log-reward", action="store_true", help="Use log2 reward shaping instead of merge sum")
    parser.add_argument("--eval-every", type=int, default=50_000, help="Evaluate policy every N environment steps")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--log-interval", type=int, default=5_000, help="Print training stats every N steps")
    parser.add_argument("--spawn-b-prob", type=float, default=0.1, help="Probability of spawning a level-2 tile (B)")
    parser.add_argument("--warmup-steps", type=int, default=20_000, help="Replay warmup before updates begin")
    parser.add_argument("--use-prioritized", action="store_true", help="Enable prioritized replay and beta annealing")
    parser.add_argument(
        "--dueling/--no-dueling",
        dest="dueling",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Toggle dueling architecture",
    )
    parser.add_argument("--save-every", type=int, default=100_000, help="Save checkpoint every N steps (in addition to best)")
    parser.add_argument(
        "--curriculum",
        dest="curriculum",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable curriculum environment reset density schedule",
    )
    parser.add_argument(
        "--curriculum-type",
        type=str,
        default="adaptive",
        choices=("linear", "exponential", "adaptive"),
        help="Curriculum schedule type",
    )
    parser.add_argument("--initial-density", type=int, default=2, help="Initial prefilled tile density for curriculum")
    parser.add_argument("--max-density", type=int, default=12, help="Maximum prefilled tile density for curriculum")
    parser.add_argument("--use-reward-shaping", action="store_true", help="Enable auxiliary reward shaping")
    parser.add_argument("--merge-bonus", type=float, default=0.05, help="Coefficient for merge-chain bonus")
    parser.add_argument("--entropy-penalty", type=float, default=0.1, help="Coefficient for entropy penalty")
    parser.add_argument(
        "--merge-preservation",
        type=float,
        default=0.01,
        help="Coefficient for preserving future merge opportunities",
    )
    return parser.parse_args()


def evaluate_policy(
    agent: DQNAgent,
    episodes: int,
    *,
    base_seed: int,
    env_config: Alphabet2048Config,
    reward_shaper: RewardShaper | None = None,
) -> Dict[str, float | Counter[str]]:
    rewards: List[float] = []
    lengths: List[int] = []
    largest_levels: List[int] = []
    invalid_counts: List[int] = []

    for idx in range(episodes):
        eval_env = Alphabet2048Env(env_config, reward_shaper=reward_shaper)
        state, info = eval_env.reset(seed=base_seed + idx)
        done = False
        total_reward = 0.0
        steps = 0
        invalid_moves = 0
        valid_mask = info["valid_actions"].astype(bool)
        while not done:
            action = agent.select_action(
                state,
                valid_mask,
                train_step=agent.config.epsilon_decay_steps,
                explore=False,
            )
            next_state, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            if info.get("invalid_action"):
                invalid_moves += 1
            state = next_state
            valid_mask = info.get("valid_actions", np.ones(utils.NUM_ACTIONS, dtype=bool)).astype(bool)
        rewards.append(total_reward)
        lengths.append(steps)
        largest_levels.append(info.get("largest_level", 0))
        invalid_counts.append(invalid_moves)
        eval_env.close()

    dist = Counter(utils.int_to_letter(level) or "-" for level in largest_levels)
    mean_return = float(statistics.mean(rewards)) if rewards else 0.0
    median_return = float(statistics.median(rewards)) if rewards else 0.0
    mean_length = float(statistics.mean(lengths)) if lengths else 0.0
    invalid_rate = float(statistics.mean(i / max(l, 1) for i, l in zip(invalid_counts, lengths))) if lengths else 0.0

    return {
        "mean_return": mean_return,
        "median_return": median_return,
        "mean_length": mean_length,
        "invalid_rate": invalid_rate,
        "largest_distribution": dist,
    }


def main() -> None:
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    utils.seed_everything(args.seed)

    reward_shaper = None
    if args.use_reward_shaping:
        reward_shaper = RewardShaper(
            use_merge_bonus=True,
            use_entropy_penalty=True,
            use_merge_preservation=True,
            merge_bonus_coef=args.merge_bonus,
            entropy_penalty_coef=args.entropy_penalty,
            merge_preservation_coef=args.merge_preservation,
        )

    env_config = Alphabet2048Config(
        seed=args.seed,
        log_reward=args.log_reward,
        spawn_b_probability=args.spawn_b_prob,
    )
    if args.curriculum:
        curriculum_config = CurriculumConfig(
            initial_board_density=args.initial_density,
            max_board_density=args.max_density,
            schedule_type=args.curriculum_type,
        )
        train_env = CurriculumEnv(env_config, curriculum_config, reward_shaper=reward_shaper)
    else:
        train_env = Alphabet2048Env(env_config, reward_shaper=reward_shaper)

    agent_config = DQNConfig(
        board_size=train_env.board_size,
        warmup_steps=args.warmup_steps,
        device=args.device,
        epsilon_decay_steps=args.total_steps,
        use_dueling=args.dueling,
        use_prioritized=args.use_prioritized,
    )
    agent = DQNAgent(agent_config)

    step = 0
    episode_idx = 0
    best_mean_return = float("-inf")
    rolling_returns: deque[float] = deque(maxlen=100)

    start_time = time.perf_counter()
    last_log_time = start_time
    last_log_step = 0
    latest_loss = None
    metrics_history: List[Dict[str, float]] = []

    state, info = train_env.reset(seed=args.seed)
    valid_mask = info["valid_actions"].astype(bool)
    episode_return = 0.0
    episode_invalid = 0
    episode_steps = 0

    while step < args.total_steps:
        action = agent.select_action(state, valid_mask, train_step=step, explore=True)
        next_state, reward, terminated, truncated, info = train_env.step(action)
        done = terminated or truncated
        next_valid_mask = info.get("valid_actions", np.ones(utils.NUM_ACTIONS, dtype=bool)).astype(bool)

        transition: Transition = (
            state.copy(),
            int(action),
            float(reward),
            next_state.copy(),
            bool(done),
            next_valid_mask.copy(),
        )
        update_info = agent.update([transition])
        if update_info is not None:
            latest_loss = update_info["loss"]

        state = next_state
        valid_mask = next_valid_mask
        episode_return += reward
        episode_steps += 1
        episode_invalid += int(info.get("invalid_action", False))

        step += 1

        if done:
            rolling_returns.append(episode_return)
            elapsed = time.perf_counter() - start_time
            metrics_history.append(
                {
                    "episode": float(episode_idx),
                    "steps": float(step),
                    "return": float(episode_return),
                    "largest_level": float(info.get("largest_level", 0)),
                    "invalid_rate": float(episode_invalid / max(episode_steps, 1)),
                }
            )
            episode_idx += 1
            state, info = train_env.reset(seed=args.seed + episode_idx)
            valid_mask = info["valid_actions"].astype(bool)
            episode_return = 0.0
            episode_invalid = 0
            episode_steps = 0

        if args.log_interval and step % args.log_interval == 0:
            now = time.perf_counter()
            delta_t = max(now - last_log_time, 1e-6)
            delta_steps = step - last_log_step
            steps_per_sec = delta_steps / delta_t
            avg_return = statistics.mean(rolling_returns) if rolling_returns else 0.0
            loss_val = float(latest_loss) if latest_loss is not None else float("nan")
            curriculum_suffix = ""
            if args.curriculum and isinstance(train_env, CurriculumEnv):
                curriculum_suffix = f" curriculum_density={train_env.curriculum.current_density()}"
            print(
                f"step={step:,} eps={agent._epsilon(step):.3f} loss={loss_val:.4f} "
                f"avg_return={avg_return:.1f} steps/s={steps_per_sec:.1f}{curriculum_suffix}"
            )
            last_log_time = now
            last_log_step = step

        if args.eval_every and step % args.eval_every == 0:
            eval_metrics = evaluate_policy(
                agent,
                args.eval_episodes,
                base_seed=args.seed + 1337 + step,
                env_config=env_config,
                reward_shaper=reward_shaper,
            )
            mean_return = eval_metrics["mean_return"]
            print(
                f"[eval] step={step:,} mean_return={mean_return:.1f} median={eval_metrics['median_return']:.1f} "
                f"len={eval_metrics['mean_length']:.1f} invalid_rate={eval_metrics['invalid_rate']:.2%}"
            )
            if isinstance(eval_metrics["largest_distribution"], Counter):
                top = eval_metrics["largest_distribution"].most_common(3)
                summary = ", ".join(f"{letter}:{count}" for letter, count in top)
                print(f"[eval] largest tiles: {summary}")

            ckpt_path = args.checkpoint_dir / f"step_{step}.pt"
            agent.save(str(ckpt_path))

            if mean_return > best_mean_return:
                best_mean_return = mean_return
                best_path = args.checkpoint_dir / "best.pt"
                agent.save(str(best_path))
                print(f"[checkpoint] new best saved to {best_path}")

            if mean_return >= args.target_avg_return:
                print("Target return reached, stopping training.")
                break

        if args.save_every and step % args.save_every == 0:
            periodic_path = args.checkpoint_dir / f"periodic_{step}.pt"
            agent.save(str(periodic_path))

    final_path = args.checkpoint_dir / "final.pt"
    agent.save(str(final_path))
    print(f"Training complete. Final checkpoint saved to {final_path}")

    log_path = args.checkpoint_dir / "training_metrics.json"
    with log_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_history, fp, indent=2)
    print(f"Wrote training metrics to {log_path}")

    train_env.close()


if __name__ == "__main__":
    main()
