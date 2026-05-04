# Taxi-v3 — Reinforcement Learning

Portfolio 2 (RL). Tabular Q-learning and SARSA implemented from scratch on
Gymnasium's `Taxi-v3`, compared against a uniform-random baseline and a
wall-unaware Manhattan-shortest-path heuristic. The agent must learn to
navigate a 5×5 grid (with walls), pick up a passenger from one of four
fixed locations, and drop them off at another — a multi-step decision
problem with 500 discrete states and 6 actions.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows PowerShell
pip install -r requirements.txt
```

## Usage

Train:
```bash
python -m src.train --agent qlearning --config experiments/qlearning_default.yaml --episodes 5000
python -m src.train --agent sarsa     --config experiments/sarsa_default.yaml     --episodes 5000
```

Evaluate (greedy, epsilon=0):
```bash
python -m src.evaluate --agent qlearning --weights results/qlearning_default/qtable.npy --episodes 100
python -m src.evaluate --agent sarsa     --weights results/sarsa_default/qtable.npy     --episodes 100
python -m src.evaluate --agent random   --episodes 100
python -m src.evaluate --agent heuristic --episodes 100
```

Plot reward curves:
```bash
python -m src.plot_results --runs results/qlearning_default results/sarsa_default
```

Render the learned policy as a grid panel (V(s) heatmap + greedy arrows):
```bash
python -m src.plot_policy --weights results/qlearning_default/qtable.npy --output results/qlearning_default/policy.png
```

Watch an episode:
```bash
python -m src.evaluate --agent qlearning --weights results/qlearning_default/qtable.npy --episodes 1 --render
```

## Project structure

```
.
├── src/
│   ├── env.py                # Taxi-v3 wrapper + state decoder
│   ├── agents/
│   │   ├── base.py
│   │   ├── random_agent.py
│   │   ├── heuristic_agent.py
│   │   ├── q_learning.py     # tabular Q-learning, from scratch
│   │   └── sarsa.py          # tabular SARSA, from scratch
│   ├── train.py
│   ├── evaluate.py
│   ├── plot_results.py
│   └── plot_policy.py
├── experiments/              # YAML hyperparameter configs
├── results/                  # logs, Q-tables, plots
├── tests/
└── report/
```

## MDP

- **States** (500 discrete): `((taxi_row * 5 + taxi_col) * 5 + passenger_loc) * 4 + destination`.
  Decode with `src.env.decode_state(s) -> (row, col, passenger_loc, destination)`.
  `passenger_loc`: 0=R, 1=G, 2=Y, 3=B, 4=in_taxi. `destination`: 0=R, 1=G, 2=Y, 3=B.
  Locations: R(0,0), G(0,4), Y(4,0), B(4,3).
- **Actions** (6): 0=south, 1=north, 2=east, 3=west, 4=pickup, 5=dropoff.
- **Rewards**: −1 per step, **−10** illegal pickup/dropoff, **+20** successful delivery (terminal).

## Reproducibility

Every script accepts `--seed`. Training writes per-episode return, length,
illegal-action count, and delivery flag to `results/<run>/log.csv`.
Hyperparameters live in `experiments/*.yaml`.
