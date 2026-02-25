"""Diagnostics helpers for analyzing agent decisions and failures."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class DecisionPoint:
    """A single decision made by the policy during evaluation."""

    board_state: np.ndarray
    valid_actions: list[int]
    chosen_action: int
    q_values: np.ndarray
    confidence: float
    episode_return_so_far: float
    steps_into_episode: int
    game_outcome: str | None = None


class DiagnosticAnalyzer:
    """Aggregates decisions and outputs actionable summaries."""

    def __init__(self) -> None:
        self.decision_log: list[DecisionPoint] = []
        self.failure_patterns: dict[str, int] = defaultdict(int)
        self.success_patterns: dict[str, int] = defaultdict(int)

    def log_decision(self, decision: DecisionPoint) -> None:
        self.decision_log.append(decision)

    def set_episode_outcome(self, episode_start: int, outcome: str) -> None:
        for idx in range(episode_start, len(self.decision_log)):
            self.decision_log[idx].game_outcome = outcome

    def analyze_failure_modes(self) -> dict:
        if not self.decision_log:
            return {"total_decisions": 0}

        high_conf_failures = []
        low_conf_successes = []
        for idx, d in enumerate(self.decision_log):
            board_complexity = int(np.count_nonzero(d.board_state))
            if d.game_outcome == "lost" and d.confidence > 1.2:
                high_conf_failures.append(
                    {
                        "step": idx,
                        "chosen_action": int(d.chosen_action),
                        "confidence": float(d.confidence),
                        "board_complexity": board_complexity,
                    }
                )
            if d.game_outcome == "won" and d.confidence < 1.0:
                low_conf_successes.append(
                    {
                        "step": idx,
                        "chosen_action": int(d.chosen_action),
                        "confidence": float(d.confidence),
                        "board_complexity": board_complexity,
                    }
                )

        return {
            "total_decisions": len(self.decision_log),
            "high_confidence_failures": high_conf_failures,
            "low_confidence_successes": low_conf_successes,
        }

    def extract_heuristic_rules(self) -> list[str]:
        if not self.decision_log:
            return []

        rules = []
        endgame = [d for d in self.decision_log if int(np.max(d.board_state)) >= 20]
        if endgame:
            conf = [d.confidence for d in endgame]
            rules.append(f"Endgame mean confidence: {float(np.mean(conf)):.3f} over {len(endgame)} decisions")

        sparse = [d for d in self.decision_log if np.count_nonzero(d.board_state) <= 4]
        if sparse:
            chosen = [d.chosen_action for d in sparse]
            most_common = int(np.bincount(chosen, minlength=4).argmax())
            rules.append(f"Sparse-board preferred action: {most_common}")
        return rules

    def compute_regret(self) -> dict:
        if not self.decision_log:
            return {"mean_regret": 0.0, "max_regret": 0.0, "high_regret_count": 0}

        regrets = []
        for d in self.decision_log:
            valid = np.array(d.valid_actions, dtype=np.int64)
            if valid.size == 0:
                continue
            q = d.q_values.astype(np.float32)
            best_q = float(np.max(q[valid]))
            chosen_q = float(q[int(d.chosen_action)])
            regrets.append(max(0.0, best_q - chosen_q))

        if not regrets:
            return {"mean_regret": 0.0, "max_regret": 0.0, "high_regret_count": 0}

        arr = np.array(regrets, dtype=np.float32)
        return {
            "mean_regret": float(np.mean(arr)),
            "max_regret": float(np.max(arr)),
            "high_regret_count": int(np.sum(arr > 0.5)),
        }

    def save_analysis(self, filepath: str | Path) -> None:
        payload = {
            "failure_modes": self.analyze_failure_modes(),
            "heuristic_rules": self.extract_heuristic_rules(),
            "regret_analysis": self.compute_regret(),
            "total_steps": len(self.decision_log),
        }
        path = Path(filepath)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


class ComparisonAnalyzer:
    """Compare policy action against a reference expert strategy."""

    def __init__(self, expert_strategy: Callable[[np.ndarray], int] | None = None) -> None:
        self.expert_strategy = expert_strategy or self._default_strategy

    @staticmethod
    def _default_strategy(board: np.ndarray) -> int:
        del board
        return 1  # prefer right in this action encoding (0=up,1=right,2=down,3=left)

    def compare_with_expert(self, agent_action: int, board: np.ndarray) -> dict:
        expert_action = int(self.expert_strategy(board))
        return {
            "agrees_with_expert": bool(agent_action == expert_action),
            "agent_action": int(agent_action),
            "expert_action": expert_action,
            "board_complexity": float(np.count_nonzero(board)),
        }
