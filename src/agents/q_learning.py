"""Tabular Q-learning, implemented from scratch.

Update rule (Watkins, 1989):
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

Off-policy: bootstraps with the GREEDY action at the next state, regardless
of which action the (epsilon-greedy) behaviour policy will actually take.
"""
from __future__ import annotations

import random

import numpy as np

from .base import Agent


class QLearningAgent(Agent):
    def __init__(
        self,
        n_states: int = 500,
        n_actions: int = 6,
        alpha: float = 0.5,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 2000,
        seed: int | None = None,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self._episode = 0
        self.rng = random.Random(seed)
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    def select_action(self, state: int, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)
        return self._argmax(self.Q[state])

    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int | None = None,
               done: bool = False) -> None:
        q_sa = self.Q[state, action]
        target = reward
        if not done:
            target += self.gamma * self.Q[next_state].max()
        self.Q[state, action] = q_sa + self.alpha * (target - q_sa)

    def end_episode(self) -> None:
        self._episode += 1
        frac = min(1.0, self._episode / max(1, self.epsilon_decay_episodes))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _argmax(self, values: np.ndarray) -> int:
        # Random tie-breaking so all ties get explored equally.
        max_v = values.max()
        candidates = np.flatnonzero(values == max_v)
        return int(self.rng.choice(candidates.tolist()))

    def save(self, path: str) -> None:
        np.save(path, self.Q)

    def load(self, path: str) -> None:
        self.Q = np.load(path)
