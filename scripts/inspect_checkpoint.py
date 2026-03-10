#!/usr/bin/env python3
"""Inspect a saved DQN checkpoint without running evaluation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def inspect_checkpoint(path: Path) -> None:
      """Load and display information about a checkpoint file."""
      if not path.exists():
                print(f"Error: checkpoint not found: {path}", file=sys.stderr)
                sys.exit(1)

      print(f"Loading checkpoint: {path}")
      checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    print("\n=== Checkpoint Contents ===")
    for key in checkpoint:
              value = checkpoint[key]
              if isinstance(value, torch.Tensor):
                            print(f"  {key}: Tensor{tuple(value.shape)} dtype={value.dtype}")
elif isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} keys")
else:
            print(f"  {key}: {value!r}")

    # Print config if available
      if "config" in checkpoint:
                cfg = checkpoint["config"]
                print("\n=== Training Config ===")
                for k, v in (cfg.__dict__ if hasattr(cfg, "__dict__") else cfg).items():
                              print(f"  {k}: {v}")

            # Print training state if available
            for state_key in ("train_step", "episode", "epsilon", "best_eval_score"):
                      if state_key in checkpoint:
                                    print(f"\n  {state_key}: {checkpoint[state_key]}")


def main() -> None:
      parser = argparse.ArgumentParser(description="Inspect a saved DQN checkpoint")
      parser.add_argument("checkpoint", type=Path, help="Path to checkpoint .pt file")
      args = parser.parse_args()
      inspect_checkpoint(args.checkpoint)


if __name__ == "__main__":
      main()
