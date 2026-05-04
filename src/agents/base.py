"""Common agent interface."""
from __future__ import annotations


class Agent:
    n_actions: int = 6

    def select_action(self, state: int, greedy: bool = False) -> int:
        raise NotImplementedError

    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int | None, done: bool) -> None:
        return None

    def end_episode(self) -> None:
        return None

    def save(self, path: str) -> None:
        return None

    def load(self, path: str) -> None:
        return None
