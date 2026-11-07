"""MAPPO for simple_world_comm.

- One actor per role (leader, adversary, good); agents of the same role share it and
  get a one-hot index so they can still behave differently. Actors see only their own
  observation (decentralized execution).
- One critic per team, fed the global state + episode progress + member one-hot
  (centralized training), with running value normalization.
- `critic="local"` gives IPPO instead: each agent's critic sees only its own observation
  (+ progress + member one-hot), so nothing in training is centralized.
- Teams listed in `train_teams` learn; the other team follows a fixed policy
  ("heuristic" or "random"). Training both teams at once is self-play.
Hyperparameter defaults follow the MPE settings of Yu et al. (2022).
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from worldcomm.env import GOOD, LEADER, ADVERSARY, role_of
from worldcomm.metrics import summarize
from worldcomm.policies import HeuristicPolicy
from worldcomm.vec_env import VecWorldComm

ROLES = (LEADER, ADVERSARY, GOOD)
TEAM_OF_ROLE = {LEADER: "adv", ADVERSARY: "adv", GOOD: "good"}


@dataclass
class Config:
    total_steps: int = 5_000_000
    n_envs: int = 128
    n_workers: int = 8
    train_teams: tuple = ("adv", "good")
    opponent: str = "heuristic"       # policy of a team that is not trained: heuristic | random | mix
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 7e-4
    clip: float = 0.2
    epochs: int = 10
    minibatches: int = 1
    entropy_coef: float = 0.01
    entropy_good: float | None = None  # entropy coefficient of the prey actor (None = entropy_coef)
    food_shaping: float = 0.0          # training-only bonus for prey moving toward food: c * (d - gamma * d')
    value_coef: float = 1.0
    huber_delta: float = 10.0
    max_grad_norm: float = 10.0
    hidden: int = 128
    layers: int = 2                   # hidden layers in every actor and critic
    critic: str = "central"           # central (MAPPO: global state) | local (IPPO: own observation)
    eval_every: int = 25              # iterations
    eval_episodes: int = 512
    seed: int = 0
    env_kwargs: dict = field(default_factory=dict)


def mlp(inp: int, out: int, hidden: int, out_gain: float, n_layers: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = [nn.LayerNorm(inp)]  # input feature normalization, as in the reference MAPPO
    for i in range(n_layers):
        layers += [nn.Linear(inp if i == 0 else hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden)]
    layers.append(nn.Linear(hidden, out))
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, np.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(layers[-1].weight, out_gain)
    return nn.Sequential(*layers)


class ValueNorm:
    """Running mean/std of value targets (Welford); critic predicts normalized values."""

    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x: np.ndarray):
        b_mean, b_var, b_count = float(x.mean()), float(x.var()), x.size
        delta, tot = b_mean - self.mean, self.count + b_count
        self.mean += delta * b_count / tot
        self.var = (self.var * self.count + b_var * b_count + delta**2 * self.count * b_count / tot) / tot
        self.count = tot

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var) + 1e-8)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


class Agents:
    """Role layout + networks; also used to act from checkpoints."""

    def __init__(self, agents, obs_dims, act_dims, state_dim, hidden, roles=ROLES, n_layers=2):
        self.agents = list(agents)
        self.members = {r: [a for a in agents if role_of(a) == r] for r in ROLES}
        self.roles = [r for r in roles if self.members[r]]
        self.obs_dims, self.act_dims, self.state_dim, self.hidden = obs_dims, act_dims, state_dim, hidden
        self.n_layers = n_layers
        self.actors = nn.ModuleDict({
            r: mlp(obs_dims[self.members[r][0]] + self.n_id(r), act_dims[self.members[r][0]], hidden, 0.01, n_layers)
            for r in self.roles
        })

    def n_id(self, role) -> int:
        return len(self.members[role]) if len(self.members[role]) > 1 else 0

    def actor_input(self, role, obs: dict) -> torch.Tensor:
        """Stack observations of a role's members -> (n_envs, n_members, obs+id)."""
        x = np.stack([obs[a] for a in self.members[role]], axis=1)
        if self.n_id(role):
            ids = np.broadcast_to(np.eye(self.n_id(role), dtype=np.float32), (x.shape[0],) + (self.n_id(role),) * 2)
            x = np.concatenate([x, ids], axis=-1)
        return torch.as_tensor(x, dtype=torch.float32)

    @torch.no_grad()
    def act(self, role, obs: dict, deterministic=False):
        logits = self.actors[role](self.actor_input(role, obs))
        dist = Categorical(logits=logits)
        a = logits.argmax(-1) if deterministic else dist.sample()
        return a.numpy(), dist.log_prob(a).numpy()

    def state_dict(self):
        return {"actors": self.actors.state_dict(), "agents": self.agents, "obs_dims": self.obs_dims,
                "act_dims": self.act_dims, "state_dim": self.state_dim, "hidden": self.hidden, "roles": self.roles,
                "n_layers": self.n_layers}


def load_actor_policy(path: str, deterministic: bool = False):
    """Callable obs dict -> action dict for the roles stored in a checkpoint."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ag = Agents(ck["agents"], ck["obs_dims"], ck["act_dims"], ck["state_dim"], ck["hidden"], ck["roles"],
                ck.get("n_layers", 2))
    ag.actors.load_state_dict(ck["actors"])
    ag.actors.eval()

    def policy(obs: dict) -> dict:
        batch = {a: np.asarray(o, dtype=np.float32)[None] for a, o in obs.items()}
        out = {}
        for r in ag.roles:
            if not all(a in batch for a in ag.members[r]):
                continue
            acts, _ = ag.act(r, batch, deterministic)
            out.update({a: int(acts[0, i]) for i, a in enumerate(ag.members[r])})
        missing = sorted(set(obs) - set(out))
        if missing:
            raise ValueError(f"checkpoint {path} has no policy for {missing} (trained roles: {ag.roles})")
        return out

    return policy


class FixedTeam:
    """Batched random/heuristic actions for the team that does not learn."""

    def __init__(self, kind, agents, act_dims, env_kwargs, seed):
        self.kind, self.agents, self.act_dims = kind, agents, act_dims
        counts = {k: v for k, v in env_kwargs.items() if k != "max_cycles"}
        self.heur = HeuristicPolicy(seed=seed, **counts)
        self.rng = np.random.default_rng(seed)
        self.use_random: np.ndarray = np.zeros(0, dtype=bool)  # 'mix': per-env opponent choice

    def new_episodes(self, n: int):
        """'mix': each env gets a random or heuristic opponent for the coming episode."""
        self.use_random = np.asarray(self.rng.random(n) < 0.5, dtype=bool)

    def __call__(self, obs: dict) -> dict:
        n = next(iter(obs.values())).shape[0]
        heur = {}
        if self.kind != "random":
            heur = {a: np.array([self.heur.act(a, obs[a][k]) for k in range(n)]) for a in self.agents}
            if self.kind == "heuristic":
                return heur
        rand = {a: self.rng.integers(self.act_dims[a], size=n) for a in self.agents}
        if self.kind == "random":
            return rand
        if self.use_random.shape[0] != n:
            self.new_episodes(n)
        return {a: np.where(self.use_random, rand[a], heur[a]) for a in self.agents}


class MAPPO:
    def __init__(self, cfg: Config, out_dir: str | Path):
        self.cfg = cfg
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.set_num_threads(4)

        self.venv = VecWorldComm(cfg.n_envs, cfg.n_workers, cfg.env_kwargs)
        v = self.venv
        self.T = v.max_cycles
        trained_roles = [r for r in ROLES if TEAM_OF_ROLE[r] in cfg.train_teams]
        self.ag = Agents(v.agents, v.obs_dims, v.act_dims, v.state_dim, cfg.hidden, trained_roles, cfg.layers)
        self.teams = {t: [a for a in v.agents if TEAM_OF_ROLE[role_of(a)] == t] for t in ("adv", "good")}
        assert cfg.critic in ("central", "local"), cfg.critic
        if cfg.critic == "local":  # one critic per team, members' observations must have the same size
            for t in cfg.train_teams:
                assert len({v.obs_dims[a] for a in self.teams[t]}) == 1, t
        crit_in = {t: (v.state_dim if cfg.critic == "central" else v.obs_dims[self.teams[t][0]])
                   for t in cfg.train_teams}
        self.critics = nn.ModuleDict({
            t: mlp(crit_in[t] + 1 + len(self.teams[t]), 1, cfg.hidden, 1.0, cfg.layers) for t in cfg.train_teams
        })
        self.vnorm = {t: ValueNorm() for t in cfg.train_teams}
        params = list(self.ag.actors.parameters()) + list(self.critics.parameters())
        self.opt = torch.optim.Adam(params, lr=cfg.lr, eps=1e-5)
        fixed = [a for a in v.agents if TEAM_OF_ROLE[role_of(a)] not in cfg.train_teams]
        self.fixed = FixedTeam(cfg.opponent, fixed, v.act_dims, cfg.env_kwargs, cfg.seed) if fixed else None
        self.idx = {a: i for i, a in enumerate(v.agents)}
        counts = {k: v for k, v in cfg.env_kwargs.items() if k != "max_cycles"}
        self.food_slice = HeuristicPolicy(**counts).food
        self.iteration, self.env_steps, self.episode_seed = 0, 0, cfg.seed * 10_000_000

    # ---------------------------------------------------------------- rollout
    def critic_input(self, team, states, obs, t):
        """(n_envs, n_members, features): global state (MAPPO) or own observation (IPPO),
        plus episode progress and member one-hot."""
        n = states.shape[0]
        m = len(self.teams[team])
        if self.cfg.critic == "central":
            x = np.repeat(states[:, None], m, axis=1)
        else:
            x = np.stack([obs[a] for a in self.teams[team]], axis=1)
        prog = np.full((n, m, 1), t / self.T, dtype=np.float32)
        ids = np.broadcast_to(np.eye(m, dtype=np.float32), (n, m, m))
        return torch.as_tensor(np.concatenate([x, prog, ids], axis=-1), dtype=torch.float32)

    @torch.no_grad()
    def values(self, team, states, obs, t):
        v = self.critics[team](self.critic_input(team, states, obs, t)).squeeze(-1).numpy()
        return self.vnorm[team].denormalize(v)

    def collect(self, opponent=None, deterministic=False):
        """One batch of episodes (n_envs x T). Returns buffers and finished-episode stats."""
        cfg, v = self.cfg, self.venv
        seeds = self.episode_seed + np.arange(cfg.n_envs)
        self.episode_seed += cfg.n_envs
        obs, states = v.reset(seeds)
        buf: dict[str, Any] = {"obs": [], "states": [], "actions": [], "logp": [], "rewards": [],
                               "values": {t: [] for t in self.critics}}
        episodes = []
        fixed = opponent or self.fixed
        if fixed is not None:
            fixed.new_episodes(cfg.n_envs)
        for t in range(self.T):
            actions = np.zeros((cfg.n_envs, len(v.agents)), dtype=np.int64)
            logp = np.zeros_like(actions, dtype=np.float32)
            for r in self.ag.roles:
                a, lp = self.ag.act(r, obs, deterministic)
                for i, name in enumerate(self.ag.members[r]):
                    actions[:, self.idx[name]] = a[:, i]
                    logp[:, self.idx[name]] = lp[:, i]
            if fixed is not None:
                for name, a in fixed(obs).items():
                    actions[:, self.idx[name]] = a
            if opponent is None:  # evaluation only needs episode stats, not values
                for team in self.critics:
                    buf["values"][team].append(self.values(team, states, obs, t))
            buf["obs"].append(obs)
            buf["states"].append(states)
            buf["actions"].append(actions)
            buf["logp"].append(logp)
            prev = obs
            obs, states, rews, finished = v.step(actions)
            if opponent is None and cfg.food_shaping:
                rews = rews + self.shaping(prev, obs)
            buf["rewards"].append(rews)
            episodes += finished
        return buf, episodes

    def food_dist(self, o):
        return np.linalg.norm(o[:, self.food_slice].reshape(len(o), -1, 2), axis=-1).min(-1)

    def shaping(self, prev, obs):
        """Potential-based shaping for prey (potential = -c * distance to nearest food).

        Only added to training rewards; logged and evaluated returns stay the original ones.
        """
        bonus = np.zeros((self.cfg.n_envs, len(self.venv.agents)), dtype=np.float32)
        for name in self.teams["good"]:
            d0, d1 = self.food_dist(prev[name]), self.food_dist(obs[name])
            bonus[:, self.idx[name]] = self.cfg.food_shaping * (d0 - self.cfg.gamma * d1)
        return bonus

    # ---------------------------------------------------------------- update
    def advantages(self, buf, team):
        """GAE per team member; episodes end at T (no bootstrap)."""
        cols = [self.idx[a] for a in self.teams[team]]
        rew = np.stack(buf["rewards"])[:, :, cols]            # (T, E, m)
        val = np.stack(buf["values"][team])                   # (T, E, m)
        adv = np.zeros_like(rew)
        last = np.zeros_like(rew[0])
        for t in reversed(range(self.T)):
            nxt = val[t + 1] if t + 1 < self.T else 0.0
            delta = rew[t] + self.cfg.gamma * nxt - val[t]
            last = delta + self.cfg.gamma * self.cfg.gae_lambda * last * (t + 1 < self.T)
            adv[t] = last
        return adv, adv + val

    def update(self, buf):
        cfg = self.cfg
        data, old_norm = {}, {}
        for team in self.critics:
            adv, ret = self.advantages(buf, team)
            old_norm[team] = self.vnorm[team].normalize(np.stack(buf["values"][team])).reshape(-1)
            self.vnorm[team].update(ret)
            data[team] = (adv, ret)

        # flatten per role (actor) and per team (critic)
        actor_batches = {}
        for r in self.ag.roles:
            team = TEAM_OF_ROLE[r]
            members = self.ag.members[r]
            pos = [self.teams[team].index(a) for a in members]
            x = torch.cat([self.ag.actor_input(r, o) for o in buf["obs"]])  # (T*E, m, d)
            acts = np.stack(buf["actions"])[:, :, [self.idx[a] for a in members]]
            lps = np.stack(buf["logp"])[:, :, [self.idx[a] for a in members]]
            adv = data[team][0][:, :, pos]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            actor_batches[r] = (x.reshape(-1, x.shape[-1]), torch.as_tensor(acts.reshape(-1)),
                                torch.as_tensor(lps.reshape(-1)), torch.as_tensor(adv.reshape(-1), dtype=torch.float32))
        critic_batches = {}
        for team in self.critics:
            x = torch.cat([self.critic_input(team, s, o, t)
                           for t, (s, o) in enumerate(zip(buf["states"], buf["obs"]))])
            ret = self.vnorm[team].normalize(data[team][1]).reshape(-1)
            critic_batches[team] = (x.reshape(-1, x.shape[-1]), torch.as_tensor(ret, dtype=torch.float32),
                                    torch.as_tensor(old_norm[team], dtype=torch.float32))

        logs = {}
        for _ in range(cfg.epochs):
            perms = {k: torch.randperm(b[0].shape[0]) for k, b in {**actor_batches, **critic_batches}.items()}
            for mb in range(cfg.minibatches):
                loss = torch.zeros(())
                for r, (x, a, lp_old, adv) in actor_batches.items():
                    ix = perms[r].chunk(cfg.minibatches)[mb]
                    dist = Categorical(logits=self.ag.actors[r](x[ix]))
                    ratio = torch.exp(dist.log_prob(a[ix]) - lp_old[ix])
                    pg = -torch.min(ratio * adv[ix], torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv[ix]).mean()
                    ent = dist.entropy().mean()
                    coef = cfg.entropy_good if r == GOOD and cfg.entropy_good is not None else cfg.entropy_coef
                    loss = loss + pg - coef * ent
                    logs[f"pg_{r}"], logs[f"entropy_{r}"] = pg.item(), ent.item()
                for team, (x, ret, old) in critic_batches.items():
                    ix = perms[team].chunk(cfg.minibatches)[mb]
                    v = self.critics[team](x[ix]).squeeze(-1)
                    v_clip = old[ix] + torch.clamp(v - old[ix], -cfg.clip, cfg.clip)
                    hl = nn.functional.huber_loss
                    vl = torch.max(hl(v, ret[ix], delta=cfg.huber_delta, reduction="none"),
                                   hl(v_clip, ret[ix], delta=cfg.huber_delta, reduction="none")).mean()
                    loss = loss + cfg.value_coef * vl
                    logs[f"vloss_{team}"] = vl.item()
                self.opt.zero_grad()
                loss.backward()
                for team in self.critics:  # actors and critic of each team clipped separately (as in reference MAPPO)
                    actor_params = [p for r in self.ag.roles if TEAM_OF_ROLE[r] == team
                                    for p in self.ag.actors[r].parameters()]
                    nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critics[team].parameters(), cfg.max_grad_norm)
                self.opt.step()
        return logs

    # ---------------------------------------------------------------- loop
    def evaluate(self, opponent_kind: str) -> dict:
        """Trained team(s) vs a fixed opponent; in self-play each team is tested separately."""
        out = {}
        tested = self.cfg.train_teams if len(self.cfg.train_teams) == 1 else ("adv", "good")
        for team in tested:
            other = [a for a in self.venv.agents if TEAM_OF_ROLE[role_of(a)] != team]
            opp = FixedTeam(opponent_kind, other, self.venv.act_dims, self.cfg.env_kwargs, seed=123)
            keep = self.ag.roles
            self.ag.roles = [r for r in keep if TEAM_OF_ROLE[r] == team]
            eps = []
            saved_seed = self.episode_seed
            self.episode_seed = 10**9  # fixed evaluation layouts
            while len(eps) < self.cfg.eval_episodes:
                eps += self.collect(opponent=opp)[1]
            self.episode_seed = saved_seed
            self.ag.roles = keep
            s = summarize(eps)
            for k in ("adv_return", "good_return", "caught", "catches", "out_frac", "food_frac", "say_entropy"):
                out[f"eval_{team}_vs_{opponent_kind}_{k}"] = s[k]
        return out

    def save(self, name: str):
        ck = self.ag.state_dict()
        ck.update(critics=self.critics.state_dict(), config=asdict(self.cfg), env_steps=self.env_steps,
                  vnorm={t: vars(n) for t, n in self.vnorm.items()})
        torch.save(ck, self.out / name)

    def train(self):
        cfg = self.cfg
        log_path = self.out / "progress.csv"
        writer, f = None, log_path.open("w", newline="")
        t0 = time.time()
        while self.env_steps < cfg.total_steps:
            buf, episodes = self.collect()
            self.env_steps += cfg.n_envs * self.T
            self.iteration += 1
            row = {"iteration": self.iteration, "env_steps": self.env_steps, "time_s": round(time.time() - t0, 1)}
            s = summarize(episodes)
            row.update({k: s[k] for k in ("adv_return", "good_return", "caught", "catches", "out_frac", "food_frac",
                                          "hidden_frac", "say_entropy")})
            row.update(self.update(buf))
            if self.iteration % cfg.eval_every == 0 or self.env_steps >= cfg.total_steps:
                for kind in ("heuristic", "random"):
                    row.update(self.evaluate(kind))
                self.save(f"ckpt_{self.env_steps}.pt")
            if writer is None or set(row) - set(writer.fieldnames):
                # (re)open with the full header once eval columns appear
                f.close()
                old = []
                if writer:
                    with log_path.open() as rf:
                        old = list(csv.DictReader(rf))
                fields = list(dict.fromkeys([*(writer.fieldnames if writer else []), *row]))
                f = log_path.open("w", newline="")
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(old)
            writer.writerow(row)
            f.flush()
            has_eval = any(k.startswith("eval_") for k in row)
            if self.iteration % 5 == 0 or has_eval:
                msg = (f"it {self.iteration:4d} steps {self.env_steps / 1e6:5.2f}M  adv {row['adv_return']:7.2f}  "
                       f"good {row['good_return']:7.2f}  caught {row['caught']:.2f}  out {row['out_frac']:.2f}  "
                       f"say_H {row['say_entropy']:.2f}  {row['time_s']:.0f}s")
                if has_eval:
                    msg += (f"\n      eval: adv vs heur caught {row.get('eval_adv_vs_heuristic_caught', float('nan')):.2f} | "
                            f"good vs heur caught {row.get('eval_good_vs_heuristic_caught', float('nan')):.2f} "
                            f"good ret {row.get('eval_good_vs_heuristic_good_return', float('nan')):.2f}")
                print(msg, flush=True)
        f.close()
        self.save("final.pt")
        self.venv.close()
