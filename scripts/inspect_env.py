"""Explore simple_world_comm: agents, spaces, observation layout; save renders for the slides."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from worldcomm import make_env
from worldcomm.render import render_world

FIG_DIR = Path(__file__).resolve().parents[1] / "presentation" / "figures"


def describe():
    env = make_env()
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    print("agents:", env.agents)
    for a in env.agents:
        print(f"  {a:16s} obs {env.observation_space(a).shape}  act {env.action_space(a)}")
    print("global state:", env.state().shape)
    print("max_cycles:", u.max_cycles)
    w = u.world
    print("landmarks:", [l.name for l in w.landmarks])
    print("sizes: adversary", w.agents[0].size, "good", w.agents[-1].size, "forest", w.forests[0].size)
    print("max speed: adversary", w.agents[0].max_speed, "good", w.agents[-1].max_speed)
    np.set_printoptions(precision=2, suppress=True)
    print("obs[adversary_0] =", obs["adversary_0"])
    print("obs[agent_0]     =", obs["agent_0"])
    env.close()


def save(img, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(FIG_DIR / name)
    print("saved", FIG_DIR / name)


def renders():
    configs = {f"map_seed{s}.png": ({}, s) for s in range(3)}
    configs["forests_0.png"] = ({"num_forests": 0}, 1)
    configs["forests_4.png"] = ({"num_forests": 4}, 1)
    configs["agents_6v4.png"] = ({"num_good": 4, "num_adversaries": 6}, 1)
    for name, (kw, seed) in configs.items():
        env = make_env(**kw)
        env.reset(seed=seed)
        save(render_world(env.unwrapped.world, view=1.15), name)
        env.close()


if __name__ == "__main__":
    describe()
    renders()
