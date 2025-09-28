# rl-alphabet-2048

Deep Q-Network (DQN) baseline that learns to play the Alphabet 2048 puzzle on a 4×4 grid. The package bundles a Gymnasium-compatible environment, PyTorch agent, CLI tooling for training/evaluation, and a Playwright bridge that can attach a trained policy to the hosted web game.

## Environment

- **Observation space**: `Box(low=0, high=26, shape=(4, 4), dtype=int32)` where each entry is a tile *level*: `0` for empty, `1 → A`, `2 → B`, …
- **Optional encoding**: `rl2048.utils.encode_onehot(board, max_level)` returns `(max_level+1, 4, 4)` float32 planes for NN input.
- **Rewards**:
  - *Merge sum (default)* – adds the numeric value of every merged tile (e.g., merging `C` tiles yields `16` reward).
  - *Log reward* – add the resulting level increment (e.g., `+3` when `C`→`D`). Toggle via `Alphabet2048Config.log_reward=True` or `--log-reward` CLI flag.
- **Spawns**: After every valid move spawn a level-1 tile (`A`); with `spawn_b_probability` (default `0.1`) the spawn upgrades to level 2 (`B`) to mimic numeric 2048's occasional `4`.
- **Termination**: Episode ends when no actions change the board. The info dict exposes `largest_level`, `score`, `score_numeric`, `valid_actions`, and an `invalid_action` flag.

## Agent

The default [DQN configuration](rl2048/dqn_agent.py) features:

- Double DQN with soft target updates (`τ = 0.005`).
- Huber loss, Adam optimiser (`lr = 1e-4`), gradient clipping (`1.0`).
- ε-greedy exploration decaying from `1.0 → 0.05` across 500k steps, honouring invalid-action masks (invalid Q-values get `-∞`).
- Replay buffer (`capacity=200k`, `batch=1024`) with optional prioritized replay (`α=0.6`, `β: 0.4 → 1.0`) and dueling heads (toggle via CLI).
- Input encoding: one-hot planes by default; log-scaled value features when `use_onehot=False`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[test]
# Optional extras for the bridge
pip install .[web]
playwright install chromium
```

## Training

```bash
python -m rl2048.scripts.train_dqn \
  --total-steps 1000000 \
  --checkpoint-dir checkpoints \
  --eval-every 50000 \
  --eval-episodes 15 \
  --use-prioritized \
  --device cpu
```

Logs report step rate, moving average return, and periodic evaluation summaries. Checkpoints are written to `checkpoints/` (`best.pt`, `final.pt`, and step-based snapshots). Training metrics for post-analysis land in `training_metrics.json`.

### Safety & stability

- Invalid actions incur a configurable penalty (default `-1`) and never spawn a tile.
- Invalid-action masking keeps the policy from issuing dead inputs in both training and deployment.
- Use a local clone of the Alphabet 2048 webpage if CORS or DOM changes break scraping; the bridge exposes the target URL as a flag.

## Evaluation

```bash
python -m rl2048.scripts.eval_dqn \
  --checkpoint checkpoints/best.pt \
  --episodes 50 \
  --target-letter K \
  --output eval_summary.json
```

The script prints mean/median returns, average episode length, invalid-action rate, success rate for reaching a target letter, and produces a JSON report containing per-episode summaries plus a histogram of largest tiles.

## Driving the live game

```bash
python -m rl2048.scripts.web_bridge_dom \
  --checkpoint checkpoints/best.pt \
  --url https://roguetex.github.io/2A1BGame/ \
  --delay 0.12 \
  --headless
```

- Requires the optional `web` dependencies and a `playwright install chromium` run.
- The bridge polls the DOM to read board letters, maps them to RL levels, masks invalid actions, and feeds the greedy policy back to the page.
- If DOM parsing stops working, rerun with `--url` pointing at a local copy or extend the placeholder OCR fallback hook.

## Testing

```bash
pytest
```

`tests/test_env.py` exercises spawning, invalid actions, tile merges, terminal detection, and masking. `tests/test_agent.py` checks network output shapes, action masking, and that a single optimisation step reduces TD error on a handcrafted batch.

## Next steps

- Integrate PPO or QR-DQN agents using the same environment.
- Flesh out the OCR fallback to support canvas-based renderers.
- Add curriculum schedules for reward shaping and spawn probabilities.
