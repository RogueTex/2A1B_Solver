"""Utility helpers for the Alphabet 2048 reinforcement-learning toolkit."""
from __future__ import annotations

import math
import random
from typing import Iterable, Optional, Tuple

import numpy as np

try:  # Optional torch dependency for seeding
    import torch
except ImportError:  # pragma: no cover - torch may be absent in minimal setups
    torch = None  # type: ignore

BOARD_SIZE = 4
NUM_ACTIONS = 4
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTER_TO_INT = {letter: idx + 1 for idx, letter in enumerate(ALPHABET)}
INT_TO_LETTER = {idx + 1: letter for idx, letter in enumerate(ALPHABET)}


def letter_to_int(letter: str) -> int:
    """Convert a tile letter to its level encoding (A→1, …, empty→0)."""

    if not letter:
        return 0
    upper = letter.strip().upper()
    if not upper:
        return 0
    if upper not in LETTER_TO_INT:
        raise ValueError(f"Unsupported letter: {letter}")
    return LETTER_TO_INT[upper]


def int_to_letter(level: int) -> str:
    """Inverse mapping of :func:`letter_to_int`."""

    if level <= 0:
        return ""
    return INT_TO_LETTER.get(level, "")


def level_to_value(level: int | np.ndarray) -> int | np.ndarray:
    """Map a tile *level* (log2 encoding) to its numeric board value."""

    if isinstance(level, np.ndarray):
        level = level.astype(np.int64)
        values = np.zeros_like(level, dtype=np.int64)
        mask = level > 0
        values[mask] = np.left_shift(1, level[mask])
        return values
    if level <= 0:
        return 0
    return 1 << int(level)


def value_to_level(value: int | np.ndarray) -> int | np.ndarray:
    """Inverse of :func:`level_to_value`. ``0`` maps to ``0``."""

    if isinstance(value, np.ndarray):
        value = value.astype(np.float64)
        result = np.zeros_like(value, dtype=np.int64)
        mask = value > 0
        result[mask] = np.rint(np.log2(value[mask])).astype(np.int64)
        return result
    if value <= 0:
        return 0
    return int(round(math.log2(value)))


def encode_onehot(board: np.ndarray, max_level: int = 16) -> np.ndarray:
    """One-hot encode the board into ``(max_level+1, 4, 4)`` planes."""

    board = np.asarray(board, dtype=np.int32)
    planes = np.zeros((max_level + 1, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    clipped = np.clip(board, 0, max_level)
    for level in range(max_level + 1):
        planes[level] = (clipped == level).astype(np.float32)
    return planes


def _compact_line(line: Iterable[int]) -> list[int]:
    return [value for value in line if value != 0]


def _merge_line(line: np.ndarray) -> Tuple[np.ndarray, int, bool, list[int]]:
    """Slide a length-4 line to the left, performing merges once per tile."""

    dense = _compact_line(line)
    merged = []
    reward = 0
    skip = False
    merged_levels: list[int] = []
    for idx, value in enumerate(dense):
        if skip:
            skip = False
            continue
        if idx + 1 < len(dense) and dense[idx + 1] == value:
            new_level = value + 1
            merged.append(new_level)
            reward += level_to_value(new_level)
            merged_levels.append(new_level)
            skip = True
        else:
            merged.append(value)
    new_line = np.zeros(BOARD_SIZE, dtype=np.int32)
    new_line[: len(merged)] = merged
    changed = not np.array_equal(new_line, line)
    return new_line, int(reward), bool(changed), merged_levels


def slide_and_merge(board: np.ndarray, action: int) -> Tuple[np.ndarray, int, bool, list[int]]:
    """Apply a 2048 move and return ``(new_board, reward, changed, merges)``."""

    board = np.asarray(board, dtype=np.int32)
    if board.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError("board must be shape (4, 4)")
    if action not in range(NUM_ACTIONS):
        raise ValueError("action must be within {0,1,2,3}")

    new_board = board.copy()
    reward = 0
    moved = False

    merged_levels: list[int] = []

    if action == 0:  # up
        for col in range(BOARD_SIZE):
            line = board[:, col]
            merged_line, line_reward, line_moved, merges = _merge_line(line)
            new_board[:, col] = merged_line
            reward += line_reward
            moved = moved or line_moved
            merged_levels.extend(merges)
    elif action == 1:  # right
        for row in range(BOARD_SIZE):
            line = board[row, ::-1]
            merged_line, line_reward, line_moved, merges = _merge_line(line)
            new_board[row, :] = merged_line[::-1]
            reward += line_reward
            moved = moved or line_moved
            merged_levels.extend(merges)
    elif action == 2:  # down
        for col in range(BOARD_SIZE):
            line = board[::-1, col]
            merged_line, line_reward, line_moved, merges = _merge_line(line)
            new_board[:, col] = merged_line[::-1]
            reward += line_reward
            moved = moved or line_moved
            merged_levels.extend(merges)
    elif action == 3:  # left
        for row in range(BOARD_SIZE):
            line = board[row, :]
            merged_line, line_reward, line_moved, merges = _merge_line(line)
            new_board[row, :] = merged_line
            reward += line_reward
            moved = moved or line_moved
            merged_levels.extend(merges)
    else:  # pragma: no cover - handled above
        raise ValueError(f"Invalid action: {action}")

    return new_board.astype(np.int32), int(reward), bool(moved), merged_levels


def mask_invalid_actions(board: np.ndarray) -> np.ndarray:
    """Boolean mask where ``True`` marks an invalid (no-op) action."""

    invalid = np.zeros(NUM_ACTIONS, dtype=bool)
    for action in range(NUM_ACTIONS):
        _, _, changed, _ = slide_and_merge(board, action)
        invalid[action] = not changed
    return invalid


def can_move(board: np.ndarray) -> bool:
    """Return ``True`` if any valid move exists."""

    if np.any(board == 0):
        return True
    return not mask_invalid_actions(board).all()


def seed_everything(seed: Optional[int]) -> None:
    """Seed Python ``random``, NumPy, and PyTorch (if available)."""

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - environment specific
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
