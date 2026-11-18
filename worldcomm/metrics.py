"""Per-episode metrics for simple_world_comm beyond the raw rewards.

`EpisodeStats` reads the simulator state after every step, so it works for any
policies (random, heuristic, trained) and does not change what the agents see.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from worldcomm.env import LEADER, N_MOVES, N_SAY, is_adversary_team, role_of


def _collide(a, b) -> bool:
    return float(np.linalg.norm(a.state.p_pos - b.state.p_pos)) < a.size + b.size


class EpisodeStats:
    """Accumulates one episode; call `reset(env)`, `step(env, actions, rewards)`, `finish()`."""

    def __init__(self):
        self.returns: dict[str, float] = {}
        self.n_good = 0
        self.steps = 0
        self.catches = self.food_steps = self.out_steps = self.hidden_steps = 0
        self.first_catch: int | None = None
        self.say_counts = np.zeros(N_SAY, dtype=int)

    def reset(self, env):
        world = env.unwrapped.world
        self.returns = {a.name: 0.0 for a in world.agents}
        self.n_good = sum(not a.adversary for a in world.agents)
        self.steps = 0
        self.catches = 0          # (step, good agent) pairs touched by >= 1 adversary
        self.food_steps = 0       # (step, good agent) pairs on a food
        self.out_steps = 0        # (step, good agent) pairs outside the [-1, 1]^2 arena
        self.hidden_steps = 0     # (step, good agent) pairs inside a forest
        self.first_catch = None
        self.say_counts = np.zeros(N_SAY, dtype=int)

    def step(self, env, actions: dict, rewards: dict):
        world = env.unwrapped.world
        self.steps += 1
        for name, r in rewards.items():
            self.returns[name] += float(r)
        advs = [a for a in world.agents if a.adversary]
        for g in (a for a in world.agents if not a.adversary):
            if any(_collide(g, adv) for adv in advs):
                self.catches += 1
                if self.first_catch is None:
                    self.first_catch = self.steps
            self.food_steps += any(_collide(g, f) for f in world.food)
            self.out_steps += bool(np.any(np.abs(g.state.p_pos) > 1.0))
            self.hidden_steps += any(_collide(g, f) for f in world.forests)
        for name, act in actions.items():
            if role_of(name) == LEADER:
                self.say_counts[int(act) // N_MOVES] += 1

    def team_return(self, adversary: bool) -> float:
        """Mean return over the members of one team so far."""
        return float(np.mean([r for n, r in self.returns.items() if is_adversary_team(n) == adversary]))

    def finish(self) -> dict:
        denom = self.steps * self.n_good
        p = self.say_counts / max(self.say_counts.sum(), 1)
        nz = p[p > 0]
        return {
            "adv_return": self.team_return(True),     # mean over adversary team members
            "good_return": self.team_return(False),   # mean over good agents
            "catches": self.catches,
            "caught": int(self.catches > 0),
            "first_catch_step": self.first_catch,
            "food_frac": self.food_steps / denom,
            "out_frac": self.out_steps / denom,
            "hidden_frac": self.hidden_steps / denom,
            "say_entropy": float(-(nz * np.log2(nz)).sum()),  # bits, max 2
            "length": self.steps,
        }


def summarize(records: list[dict]) -> dict:
    keys = ["adv_return", "good_return", "catches", "caught", "food_frac", "out_frac", "hidden_frac", "say_entropy"]
    out: dict[str, float] = {"episodes": len(records)}
    for k in keys:
        v = np.array([r[k] for r in records], dtype=float)
        out[k] = float(v.mean())
        out[k + "_std"] = float(v.std())
    return out


def write_csv(records: list[dict], path: str | Path, **extra) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{**extra, **r} for r in records]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path
