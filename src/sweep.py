"""Hyperparameter sweep + multi-seed training driver.

Runs training in-process (no subprocess overhead) for a single agent over a
grid of values for ONE swept parameter, with multiple seeds per value.
Output layout:
    results/<sweep_name>/<param>=<value>_seed=<N>/log.csv
                                                /qtable.npy

Example:
    python -m src.sweep --agent qlearning --param alpha \
        --values 0.1 0.3 0.5 0.9 --seeds 5 --episodes 3000

Pair with `plot_sweep.py` to aggregate the runs into mean +/- std curves.
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from .agents.q_learning import QLearningAgent
from .agents.sarsa import SarsaAgent
from .env import N_ACTIONS, N_STATES, TaxiEnv


def _build_agent(name: str, agent_cfg: dict, seed: int):
    if name == "qlearning":
        return QLearningAgent(n_states=N_STATES, n_actions=N_ACTIONS,
                              seed=seed, **agent_cfg)
    if name == "sarsa":
        return SarsaAgent(n_states=N_STATES, n_actions=N_ACTIONS,
                          seed=seed, **agent_cfg)
    raise ValueError(f"Unknown agent: {name}")


def _run_one(agent_name: str, agent_cfg: dict, episodes: int, max_steps: int,
             seed: int, out_dir: Path) -> None:
    random.seed(seed)
    np.random.seed(seed)
    env = TaxiEnv()
    agent = _build_agent(agent_name, agent_cfg, seed=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["episode", "return", "length", "illegal_actions", "delivered", "epsilon"]
        )

        for ep in range(episodes):
            state = env.reset(seed=seed + ep)
            action = agent.select_action(state)
            ep_return = 0.0
            steps = 0
            illegal = 0
            delivered = 0
            for _ in range(max_steps):
                next_state, reward, done, info = env.step(action)
                ep_return += reward
                steps += 1
                if info.get("illegal_action"):
                    illegal += 1
                if info.get("delivered"):
                    delivered = 1

                if agent_name == "sarsa":
                    next_action = agent.select_action(next_state) if not done else None
                    agent.update(state, action, reward, next_state, next_action, done)
                    action = next_action if not done else 0
                else:
                    agent.update(state, action, reward, next_state, None, done)
                    action = agent.select_action(next_state) if not done else 0

                state = next_state
                if done:
                    break

            agent.end_episode()
            writer.writerow(
                [ep, f"{ep_return:.2f}", steps, illegal, delivered,
                 f"{agent.epsilon:.4f}"]
            )

    agent.save(str(out_dir / "qtable.npy"))
    env.close()


def _coerce(value: str):
    """Parse a CLI value into bool/int/float/str (in that order)."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=["qlearning", "sarsa"])
    parser.add_argument("--config", type=str,
                        default="experiments/qlearning_default.yaml",
                        help="Base config; the sweep overrides one field of `agent:`.")
    parser.add_argument("--param", required=True,
                        help="Name of the agent.<param> to sweep "
                             "(e.g. alpha, gamma, epsilon_decay_episodes).")
    parser.add_argument("--values", nargs="+", required=True,
                        help="Values for the swept parameter.")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds per value (seeds are 0..N-1).")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--sweep-name", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f) or {}
    base_agent_cfg = base_cfg.get("agent", {})

    sweep_name = args.sweep_name or f"sweep_{args.agent}_{args.param}"
    sweep_dir = Path("results") / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    values = [_coerce(v) for v in args.values]
    print(f"Sweeping {args.agent}.{args.param} over {values} with {args.seeds} seeds "
          f"x {args.episodes} episodes -> {sweep_dir}")

    total = len(values) * args.seeds
    done = 0
    start = time.time()
    for value in values:
        for seed in range(args.seeds):
            agent_cfg = deepcopy(base_agent_cfg)
            agent_cfg[args.param] = value
            run_dir = sweep_dir / f"{args.param}={value}_seed={seed}"
            _run_one(args.agent, agent_cfg, args.episodes, args.max_steps,
                     seed, run_dir)
            done += 1
            elapsed = time.time() - start
            eta = elapsed / done * (total - done)
            print(f"  [{done}/{total}] {args.param}={value} seed={seed} "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    # Save sweep manifest for reproducibility.
    with open(sweep_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(
            {
                "agent": args.agent,
                "config": args.config,
                "param": args.param,
                "values": values,
                "seeds": args.seeds,
                "episodes": args.episodes,
                "base_agent_cfg": base_agent_cfg,
            },
            f,
        )
    print(f"Done. Sweep saved to {sweep_dir}")


if __name__ == "__main__":
    main()
