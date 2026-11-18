"""Many simple_world_comm envs stepped in worker processes.

All episodes have the same fixed length (max_cycles), so the envs stay in lockstep:
`reset()` starts a new episode everywhere, then exactly `max_cycles` calls to `step()`.
Kept free of torch so spawned workers start fast.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from worldcomm.env import make_env
from worldcomm.metrics import EpisodeStats


def _worker(conn, n_envs: int, env_kwargs: dict):
    envs = [make_env(**env_kwargs) for _ in range(n_envs)]
    agents = envs[0].possible_agents
    stats = [EpisodeStats() for _ in envs]

    def pack(per_env_obs):
        return {a: np.stack([o[a] for o in per_env_obs]).astype(np.float32) for a in agents}

    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            obs_list = []
            for env, st, seed in zip(envs, stats, data):
                obs, _ = env.reset(seed=int(seed))
                st.reset(env)
                obs_list.append(obs)
            states = np.stack([env.state() for env in envs]).astype(np.float32)
            conn.send((pack(obs_list), states))
        elif cmd == "step":
            obs_list, rews, finished = [], [], []
            for k, (env, st) in enumerate(zip(envs, stats)):
                actions = {a: int(data[k, i]) for i, a in enumerate(agents)}
                obs, rew, _, _, _ = env.step(actions)
                st.step(env, actions, rew)
                obs_list.append(obs)
                rews.append([rew[a] for a in agents])
                if not env.agents:
                    finished.append(st.finish())
            states = np.stack([env.state() for env in envs]).astype(np.float32)
            conn.send((pack(obs_list), states, np.array(rews, dtype=np.float32), finished))
        elif cmd == "close":
            conn.close()
            return


class VecWorldComm:
    def __init__(self, n_envs: int, n_workers: int, env_kwargs: dict | None = None):
        env_kwargs = env_kwargs or {}
        probe = make_env(**env_kwargs)
        probe.reset(seed=0)
        self.agents = list(probe.possible_agents)
        self.obs_dims = {a: probe.observation_space(a).shape[0] for a in self.agents}
        self.act_dims = {a: probe.action_space(a).n for a in self.agents}
        self.state_dim = probe.state().shape[0]
        self.max_cycles = probe.unwrapped.max_cycles
        probe.close()

        n_workers = min(n_workers, n_envs)
        sizes = [n_envs // n_workers + (i < n_envs % n_workers) for i in range(n_workers)]
        self.sizes, self.n_envs = sizes, n_envs
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for size in sizes:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, size, env_kwargs), daemon=True)
            p.start()
            self.conns.append(parent)
            self.procs.append(p)

    def _split(self, arr):
        out, i = [], 0
        for s in self.sizes:
            out.append(arr[i:i + s])
            i += s
        return out

    def reset(self, seeds):
        for c, part in zip(self.conns, self._split(np.asarray(seeds))):
            c.send(("reset", part))
        res = [c.recv() for c in self.conns]
        obs = {a: np.concatenate([r[0][a] for r in res]) for a in self.agents}
        return obs, np.concatenate([r[1] for r in res])

    def step(self, actions: np.ndarray):
        """actions: (n_envs, n_agents) ints in `self.agents` order."""
        for c, part in zip(self.conns, self._split(actions)):
            c.send(("step", part))
        res = [c.recv() for c in self.conns]
        obs = {a: np.concatenate([r[0][a] for r in res]) for a in self.agents}
        states = np.concatenate([r[1] for r in res])
        rews = np.concatenate([r[2] for r in res])
        finished = [ep for r in res for ep in r[3]]
        return obs, states, rews, finished

    def close(self):
        for c in self.conns:
            c.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)
