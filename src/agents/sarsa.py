"""Tabular SARSA, implemented from scratch (on-policy TD control).

Update rule (Rummery & Niranjan, 1994):
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * Q(s', a') - Q(s, a) ]

On-policy: bootstraps with the action a' the (epsilon-greedy) behaviour
policy actually selects next. This makes SARSA cautious near the cliff,
because the bootstrap accounts for the agent's own random exploration.
"""
from __future__ import annotations

from .q_learning import QLearningAgent


class SarsaAgent(QLearningAgent):
    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int | None = None,
               done: bool = False) -> None:
        if not done and next_action is None:
            raise ValueError("SARSA requires next_action when not terminal.")
        q_sa = self.Q[state, action]
        target = reward
        if not done:
            target += self.gamma * self.Q[next_state, next_action]
        self.Q[state, action] = q_sa + self.alpha * (target - q_sa)
