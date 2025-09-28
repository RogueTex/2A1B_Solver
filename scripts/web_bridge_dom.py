"""Drive the hosted Alphabet 2048 game using a trained DQN agent."""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from rl2048 import utils
from rl2048.dqn_agent import DQNAgent, DQNConfig


ACTION_TO_KEY = {0: "ArrowUp", 1: "ArrowRight", 2: "ArrowDown", 3: "ArrowLeft"}


@dataclass
class WebBridgeConfig:
    url: str = "https://roguetex.github.io/2A1BGame/"
    headless: bool = True
    use_selenium: bool = False
    use_ocr_fallback: bool = False


class PlaywrightBridge:
    """Lightweight wrapper that exposes DOM helpers for the Alphabet 2048 page."""

    def __init__(self, config: WebBridgeConfig):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Playwright is not installed. Run 'pip install playwright' and 'playwright install'."
            ) from exc

        self._sync_playwright = sync_playwright
        self.config = config
        self._playwright = None
        self._browser = None
        self.page = None

    def __enter__(self) -> "PlaywrightBridge":
        self._playwright = self._sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.config.headless)
        self.page = self._browser.new_page()
        self.page.goto(self.config.url, wait_until="networkidle")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def read_board(self) -> np.ndarray:
        if self.page is None:
            raise RuntimeError("Bridge not initialised")
        if self.config.use_ocr_fallback:
            raise NotImplementedError("OCR fallback not implemented; DOM parsing should suffice for this site.")

        tiles = self.page.query_selector_all(".tile")
        board = np.zeros((4, 4), dtype=np.int32)
        for tile in tiles:
            if tile is None:
                continue
            text = (tile.inner_text() or "").strip()
            classes = (tile.get_attribute("class") or "").split()
            row, col = self._extract_position(classes, tile)
            level = self._parse_tile_level(text)
            if row is None or col is None or level is None:
                continue
            board[row, col] = level
        return board

    def send_action(self, action: int) -> None:
        if self.page is None:
            raise RuntimeError("Bridge not initialised")
        key = ACTION_TO_KEY.get(action)
        if key is None:
            raise ValueError(f"Unsupported action {action}")
        self.page.keyboard.press(key)

    def game_over(self, board: np.ndarray) -> bool:
        if self.page is None:
            raise RuntimeError("Bridge not initialised")
        overlay = self.page.locator(".game-message, .game-over, .overlay")
        if overlay.count() > 0:
            try:
                if overlay.first.is_visible():  # type: ignore[attr-defined]
                    return True
            except Exception:  # pragma: no cover - visibility check best effort
                pass
        return utils.mask_invalid_actions(board).all()

    # ------------------------------------------------------------------
    # Internal parsing helpers
    # ------------------------------------------------------------------

    def _extract_position(self, classes: Iterable[str], tile) -> tuple[Optional[int], Optional[int]]:
        for cls in classes:
            if cls.startswith("tile-position-"):
                try:
                    _, _, x_str, y_str = cls.split("-")
                    col = int(x_str) - 1
                    row = int(y_str) - 1
                    return row, col
                except ValueError:
                    continue
        style = tile.get_attribute("style") or ""
        if "translate" in style:
            try:
                coords = style.split("translate(")[-1].split(")")[0]
                x_str, y_str = [val.strip().replace("px", "") for val in coords.split(",")]
                col = max(0, min(3, round(float(x_str) / 90)))
                row = max(0, min(3, round(float(y_str) / 90)))
                return row, col
            except (ValueError, IndexError):
                return None, None
        return None, None

    def _parse_tile_level(self, text: str) -> Optional[int]:
        if not text:
            return None
        if text.isdigit():
            return int(utils.value_to_level(int(text)))
        try:
            return utils.letter_to_int(text)
        except ValueError:
            return None


def loop_agent(
    agent: DQNAgent,
    bridge: PlaywrightBridge,
    *,
    delay: float,
    max_steps: int,
) -> dict:
    steps = 0
    rewards = []
    largest_levels = []
    board = bridge.read_board()

    while steps < max_steps:
        if bridge.game_over(board):
            break
        valid_mask = ~utils.mask_invalid_actions(board)
        action = agent.select_action(
            board,
            valid_mask,
            train_step=agent.config.epsilon_decay_steps,
            explore=False,
        )
        bridge.send_action(action)
        time.sleep(delay)
        board = bridge.read_board()
        largest_levels.append(int(board.max()))
        rewards.append(float(utils.level_to_value(board).sum()))
        steps += 1

    return {
        "steps": steps,
        "largest_level": int(board.max()),
        "largest_letter": utils.int_to_letter(int(board.max())) or "-",
        "total_reward_estimate": float(rewards[-1]) if rewards else 0.0,
        "max_steps_reached": steps >= max_steps,
    }


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Drive the Alphabet 2048 web game with a trained policy")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--url", type=str, default=WebBridgeConfig.url, help="Alphabet 2048 page URL")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for inference")
    parser.add_argument("--headless", action="store_true", help="Run Playwright headless")
    parser.add_argument("--delay", type=float, default=0.15, help="Delay (seconds) between actions")
    parser.add_argument("--max-steps", type=int, default=3_000, help="Safety cap on number of actions to send")
    parser.add_argument("--stats-out", type=pathlib.Path, default=None, help="Optional JSON file to store run summary")
    parser.add_argument("--use-selenium", action="store_true", help="Use Selenium instead of Playwright (slower)")
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Attempt screenshot/OCR fallback if DOM scraping fails (not yet implemented)",
    )
    args = parser.parse_args()

    if args.use_selenium:
        raise NotImplementedError("Selenium bridge not yet wired. Use Playwright for best results.")

    bridge_config = WebBridgeConfig(
        url=args.url,
        headless=args.headless,
        use_selenium=args.use_selenium,
        use_ocr_fallback=args.use_ocr,
    )

    agent = DQNAgent(DQNConfig(device=args.device))
    agent.load(str(args.checkpoint))

    with PlaywrightBridge(bridge_config) as bridge:
        summary = loop_agent(agent, bridge, delay=args.delay, max_steps=args.max_steps)

    print("=== Run summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if args.stats_out is not None:
        with args.stats_out.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        print(f"Saved summary to {args.stats_out}")


if __name__ == "__main__":
    run_cli()
