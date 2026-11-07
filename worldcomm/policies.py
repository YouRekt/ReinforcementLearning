"""Non-learning reference policies acting from each agent's own observation.

Observation layout (default config: 1 obstacle, 2 food, 2 forests, 4 adversaries, 2 good):
  [0:2] own velocity, [2:4] own position, [4:14] landmarks relative
  (obstacle, food x2, forest x2), [14:24] other agents relative (world order
  without self; a hidden agent reads (0, 0)), then team-specific tails.
"""

from __future__ import annotations

import numpy as np

from worldcomm.env import GOOD, N_MOVES, role_of

# Unit vectors of discrete moves: stop, left, right, down, up
MOVE_DIRS = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=float)


def _best_move(direction: np.ndarray) -> int:
    if np.linalg.norm(direction) < 1e-8:
        return 0
    return int(np.argmax(MOVE_DIRS @ direction))


class RandomPolicy:
    def __init__(self, env, seed: int = 0):
        self.env = env
        for i, a in enumerate(env.possible_agents):
            env.action_space(a).seed(seed * 100 + i)

    def __call__(self, obs: dict) -> dict:
        return {a: int(self.env.action_space(a).sample()) for a in obs}


class HeuristicPolicy:
    """Adversaries chase the nearest visible good agent; good agents go to food and flee.

    The leader says nothing useful (always say_0), so it is a pure chasing baseline.
    Pass the same counts as to `make_env` so the observation offsets line up.
    """

    def __init__(self, num_adversaries=4, num_good=2, num_obstacles=1, num_food=2, num_forests=2, seed=0):
        self.n_adv, self.n_good = num_adversaries, num_good
        self.food = slice(4 + 2 * num_obstacles, 4 + 2 * (num_obstacles + num_food))
        start = 4 + 2 * (num_obstacles + num_food + num_forests)
        self.others = slice(start, start + 2 * (num_adversaries + num_good - 1))
        self.rng = np.random.default_rng(seed)

    def act(self, agent: str, o: np.ndarray) -> int:
        pos = o[2:4]
        food = o[self.food].reshape(-1, 2)
        others = o[self.others].reshape(-1, 2)
        if role_of(agent) == GOOD:
            advs = others[: self.n_adv]
            target = food[np.argmin(np.linalg.norm(food, axis=1))]
            d = target / (np.linalg.norm(target) + 1e-8)
            for rel in advs:
                dist = np.linalg.norm(rel)
                if 1e-6 < dist < 0.5:  # (0, 0) means hidden: ignore
                    d -= 2.0 * rel / dist**2 * 0.1
            d -= np.where(np.abs(pos) > 0.8, np.sign(pos), 0.0)  # stay in the arena
            return _best_move(d)
        prey = others[-self.n_good:]
        visible = [p for p in prey if np.linalg.norm(p) > 1e-6]
        if visible:
            move = _best_move(visible[int(np.argmin([np.linalg.norm(p) for p in visible]))])
        else:
            move = int(self.rng.integers(N_MOVES))
        return move  # leader: say_0 + move, i.e. the same index

    def __call__(self, obs: dict) -> dict:
        return {a: self.act(a, o) for a, o in obs.items()}


def make_policy(spec: str, env, seed: int = 0, **env_kwargs):
    """'random' | 'heuristic' | 'mappo:<checkpoint.pt>' -> callable obs dict -> action dict.

    `env_kwargs` are the counts passed to make_env (needed for observation offsets).
    """
    if spec == "random":
        return RandomPolicy(env, seed)
    if spec == "heuristic":
        return HeuristicPolicy(seed=seed, **env_kwargs)
    if spec.startswith("mappo:"):
        from worldcomm.mappo import load_actor_policy
        return load_actor_policy(spec.split(":", 1)[1])
    raise ValueError(f"unknown policy spec {spec!r}")


class TeamMix:
    """Combine two policies: one for the adversary team, one for the good agents."""

    def __init__(self, adversaries, good):
        self.adv, self.good = adversaries, good

    def __call__(self, obs: dict) -> dict:
        adv_obs = {a: o for a, o in obs.items() if role_of(a) != GOOD}
        good_obs = {a: o for a, o in obs.items() if role_of(a) == GOOD}
        return {**self.adv(adv_obs), **self.good(good_obs)}
