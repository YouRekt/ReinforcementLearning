"""Train MAPPO on simple_world_comm.

  uv run python scripts/train_mappo.py --train adv good --steps 5e6 --run selfplay_s0
  uv run python scripts/train_mappo.py --train adv --opponent heuristic --run adv_vs_heur_s0
  uv run python scripts/train_mappo.py --critic local --steps 2e7 --run ippo_h128_s0      # IPPO
  uv run python scripts/train_mappo.py --hidden 256 --layers 3 --steps 2e7 --run mappo_h256x3_s0
  uv run python scripts/train_mappo.py --num-forests 4 --max-cycles 50 --steps 2e7 --run env_f4_s0   # other env settings
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from worldcomm.mappo import MAPPO, Config

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="output folder name under runs/")
    p.add_argument("--train", nargs="+", default=["adv", "good"], choices=["adv", "good"])
    p.add_argument("--opponent", default="heuristic", choices=["heuristic", "random", "mix"])
    p.add_argument("--steps", type=float, default=5e6)
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--entropy-good", type=float, default=None, help="entropy coefficient for prey only")
    p.add_argument("--prey-vs-heuristic", type=float, default=0.0,
                   help="self-play: fraction of iterations where prey train against heuristic predators")
    p.add_argument("--food-shaping", type=float, default=0.0, help="training-only reward for prey approaching food")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--hidden", type=int, default=128, help="width of hidden layers")
    p.add_argument("--layers", type=int, default=2, help="number of hidden layers")
    p.add_argument("--critic", default="central", choices=["central", "local"], help="central = MAPPO, local = IPPO")
    p.add_argument("--num-forests", type=int, default=None, help="environment setting (default 2)")
    p.add_argument("--max-cycles", type=int, default=None, help="episode length (default 25)")
    args = p.parse_args()
    env_kwargs = {k: v for k, v in [("num_forests", args.num_forests), ("max_cycles", args.max_cycles)] if v is not None}

    cfg = Config(total_steps=int(args.steps), n_envs=args.envs, n_workers=args.workers,
                 train_teams=tuple(args.train), opponent=args.opponent, seed=args.seed, lr=args.lr,
                 entropy_coef=args.entropy, entropy_good=args.entropy_good, food_shaping=args.food_shaping, prey_vs_heuristic=args.prey_vs_heuristic, epochs=args.epochs, eval_every=args.eval_every,
                 hidden=args.hidden, layers=args.layers, critic=args.critic,
                 env_kwargs=env_kwargs)
    out = ROOT / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    MAPPO(cfg, out).train()


if __name__ == "__main__":
    main()
