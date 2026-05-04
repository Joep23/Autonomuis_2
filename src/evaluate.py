"""Evaluate an agent (greedy) over N episodes and print summary stats."""
from __future__ import annotations

import argparse
import statistics

from .agents.heuristic_agent import HeuristicAgent
from .agents.q_learning import QLearningAgent
from .agents.random_agent import RandomAgent
from .agents.sarsa import SarsaAgent
from .env import N_ACTIONS, N_STATES, TaxiEnv


def build_agent(name: str, weights: str | None, seed: int):
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "heuristic":
        return HeuristicAgent(seed=seed)
    if name in ("qlearning", "sarsa"):
        cls = QLearningAgent if name == "qlearning" else SarsaAgent
        agent = cls(n_states=N_STATES, n_actions=N_ACTIONS, seed=seed)
        if weights:
            agent.load(weights)
        agent.epsilon = 0.0
        return agent
    raise ValueError(f"Unknown agent: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True,
                        choices=["random", "heuristic", "qlearning", "sarsa"])
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = TaxiEnv(render_mode="human" if args.render else None)
    agent = build_agent(args.agent, args.weights, args.seed)

    returns: list[float] = []
    lengths: list[int] = []
    illegals: list[int] = []
    success: list[int] = []

    for ep in range(args.episodes):
        state = env.reset(seed=args.seed + ep)
        ep_return = 0.0
        steps = 0
        illegal = 0
        delivered = 0
        for _ in range(args.max_steps):
            action = agent.select_action(state, greedy=True)
            state, reward, done, info = env.step(action)
            ep_return += reward
            steps += 1
            if info.get("illegal_action"):
                illegal += 1
            if info.get("delivered"):
                delivered = 1
            if done:
                break
        returns.append(ep_return)
        lengths.append(steps)
        illegals.append(illegal)
        success.append(delivered)

    env.close()

    def stats(xs: list[float]) -> str:
        return (
            f"mean={statistics.mean(xs):.2f} "
            f"std={statistics.pstdev(xs):.2f} "
            f"min={min(xs):.2f} max={max(xs):.2f}"
        )

    print(f"Agent: {args.agent} ({args.episodes} episodes, greedy)")
    print(f"  Return         : {stats(returns)}")
    print(f"  Length         : {stats([float(l) for l in lengths])}")
    print(f"  Illegal actions: {stats([float(i) for i in illegals])}")
    print(f"  Success rate   : {sum(success) / len(success):.2%}")


if __name__ == "__main__":
    main()
