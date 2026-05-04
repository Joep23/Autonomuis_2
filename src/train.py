"""Train Q-learning or SARSA on CliffWalking and log per-episode metrics."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import yaml

from .agents.q_learning import QLearningAgent
from .agents.sarsa import SarsaAgent
from .env import N_ACTIONS, N_STATES, TaxiEnv


def build_agent(name: str, cfg: dict, seed: int):
    agent_cfg = cfg.get("agent", {})
    if name == "qlearning":
        return QLearningAgent(n_states=N_STATES, n_actions=N_ACTIONS,
                              seed=seed, **agent_cfg)
    if name == "sarsa":
        return SarsaAgent(n_states=N_STATES, n_actions=N_ACTIONS,
                          seed=seed, **agent_cfg)
    raise ValueError(f"Unknown trainable agent: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=["qlearning", "sarsa"])
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = TaxiEnv()
    agent = build_agent(args.agent, cfg, seed=args.seed)

    run_name = args.run_name or Path(args.config).stem
    out_dir = Path("results") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["episode", "return", "length", "illegal_actions", "delivered", "epsilon"]
        )

        for ep in range(args.episodes):
            state = env.reset(seed=args.seed + ep)
            action = agent.select_action(state)
            ep_return = 0.0
            steps = 0
            illegal = 0
            delivered = 0

            for _ in range(args.max_steps):
                next_state, reward, done, info = env.step(action)
                ep_return += reward
                steps += 1
                if info.get("illegal_action"):
                    illegal += 1
                if info.get("delivered"):
                    delivered = 1

                if args.agent == "sarsa":
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
            if (ep + 1) % 200 == 0:
                f.flush()
                print(f"[{run_name}] ep={ep + 1} return={ep_return:.1f} "
                      f"len={steps} illegal={illegal} delivered={delivered} "
                      f"eps={agent.epsilon:.3f}")

    agent.save(str(out_dir / "qtable.npy"))
    env.close()
    print(f"Saved Q-table and log to {out_dir}")


if __name__ == "__main__":
    main()
