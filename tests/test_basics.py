"""Smoke tests: env round-trips, agents return valid actions, learning runs."""
import numpy as np

from src.agents.heuristic_agent import HeuristicAgent
from src.agents.q_learning import QLearningAgent
from src.agents.random_agent import RandomAgent
from src.agents.sarsa import SarsaAgent
from src.env import N_ACTIONS, N_STATES, TaxiEnv, decode_state


def test_env_reset_returns_valid_state():
    env = TaxiEnv()
    s = env.reset(seed=0)
    assert isinstance(s, int) and 0 <= s < N_STATES
    env.close()


def test_env_step_returns_int_state_float_reward():
    env = TaxiEnv()
    env.reset(seed=0)
    s2, r, done, info = env.step(0)
    assert isinstance(s2, int) and 0 <= s2 < N_STATES
    assert isinstance(r, float)
    assert isinstance(done, bool)
    assert "illegal_action" in info and "delivered" in info
    env.close()


def test_decode_roundtrip():
    # Encode (taxi_row=2, taxi_col=3, pass_loc=1, dest=2) and decode back.
    state = ((2 * 5 + 3) * 5 + 1) * 4 + 2
    assert decode_state(state) == (2, 3, 1, 2)


def test_random_and_heuristic_actions_in_range():
    for agent in (RandomAgent(seed=0), HeuristicAgent(seed=0)):
        for s in (0, 100, 250, 499):
            a = agent.select_action(s)
            assert 0 <= a < N_ACTIONS


def test_qlearning_update_changes_q():
    agent = QLearningAgent(seed=0)
    before = agent.Q[100, 1]
    agent.update(state=100, action=1, reward=-1.0, next_state=120, done=False)
    assert agent.Q[100, 1] != before


def test_sarsa_update_changes_q():
    agent = SarsaAgent(seed=0)
    before = agent.Q[100, 1]
    agent.update(state=100, action=1, reward=-1.0,
                 next_state=120, next_action=0, done=False)
    assert agent.Q[100, 1] != before


def test_qtable_save_load_roundtrip(tmp_path):
    agent = QLearningAgent(seed=0)
    agent.Q = np.random.RandomState(0).randn(N_STATES, N_ACTIONS)
    path = tmp_path / "q.npy"
    agent.save(str(path))
    other = QLearningAgent(seed=0)
    other.load(str(path))
    assert np.allclose(agent.Q, other.Q)
