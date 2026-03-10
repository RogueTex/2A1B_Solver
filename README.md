[![CI](https://github.com/RogueTex/2A1B_Solver/actions/workflows/ci.yml/badge.svg)](https://github.com/RogueTex/2A1B_Solver/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# 2A1B Solver

This is my reinforcement learning solver for the Alphabet 2048-style 2A1B game.
The project includes:

- a Gymnasium environment for the game logic
- a PyTorch DQN agent (Double DQN + optional dueling/prioritized replay)
- train/eval scripts
- a web bridge to control the live browser game

I also added a core enhancement stack for better training quality:

- curriculum learning
- reward shaping
- decision diagnostics

## Project layout

```text
2A1B_Solver/
├── rl2048/
│   ├── env_alphabet2048.py
│   ├── dqn_agent.py
│   ├── curriculum.py
│   ├── reward_shaper.py
│   ├── diagnostics.py
│   └── scripts/
│       ├── train_dqn.py
│       └── eval_dqn.py
├── scripts/
│   ├── train_dqn.py
│   ├── eval_dqn.py
│   └── web_bridge_dom.py
└── tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[test]
```

For browser play:

```bash
pip install -e .[web]
playwright install chromium
```

## Train

Baseline training:

```bash
python -m rl2048.scripts.train_dqn \
  --total-steps 1000000 \
  --checkpoint-dir checkpoints \
  --eval-every 50000 \
  --eval-episodes 15 \
  --use-prioritized \
  --device cpu
```

Training with enhancements:

```bash
python -m rl2048.scripts.train_dqn \
  --total-steps 500000 \
  --curriculum \
  --curriculum-type adaptive \
  --initial-density 2 \
  --max-density 12 \
  --use-reward-shaping
```

## Evaluate

Standard eval:

```bash
python -m rl2048.scripts.eval_dqn \
  --checkpoint checkpoints/best.pt \
  --episodes 50 \
  --target-letter K \
  --output eval_summary.json
```

Eval with diagnostics output:

```bash
python -m rl2048.scripts.eval_dqn \
  --checkpoint checkpoints/best.pt \
  --episodes 50 \
  --save-diagnostics \
  --diagnostics-output diagnostics.json
```

## Play on the live game page

```bash
python -m rl2048.scripts.web_bridge_dom \
  --checkpoint checkpoints/best.pt \
  --url https://roguetex.github.io/2A1BGame/ \
  --delay 0.12 \
  --headless
```

## Tests

```bash
pytest
```

## Architecture

The project is organised into four cooperating layers:

| Layer | Module | Responsibility |
|-------|--------|----------------|
| **Environment** | `env_alphabet2048.py` | Gymnasium-compliant game logic, step/reset, invalid-move masking |
| **Agent** | `dqn_agent.py` | Double DQN, optional dueling head, optional prioritised experience replay |
| **Enhancement** | `curriculum.py`, `reward_shaper.py` | Board-density curriculum scheduling; shaped reward signal on top of base env rewards |
| **Diagnostics** | `diagnostics.py` | Decision-point logging, Q-value confidence tracking, post-hoc JSON analysis |

Data flows as follows during training:

```
Env.reset() → Board state
  → Agent.select_action() → Action
  → Env.step(action) → (next_state, base_reward, done, info)
  → RewardShaper.shape_reward() → shaped_reward
  → Agent.update() → gradient step
  → Diagnostics.log_decision() → decision record
```

## Notes

- State is a `4x4` board with values `0..26` (`0` is empty, `1` is A, ...).
- Actions use the project’s internal mapping (`0=up, 1=right, 2=down, 3=left`).
- Invalid moves are masked during inference and penalized in training.
- `rl2048/scripts/*` provides docs-style module entrypoints.
