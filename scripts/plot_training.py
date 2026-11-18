"""Learning curves of MAPPO runs against fixed opponents, with heuristic baselines.

  uv run python scripts/plot_training.py runs/selfplay_s0 runs/adv_vs_heur_s0 --out presentation/figures/training.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from worldcomm.plotting import COLORS, MUTED, style_axes

ROOT = Path(__file__).resolve().parents[1]


def baseline(adv: str, good: str, key: str) -> float:
    rows = json.loads((ROOT / "results" / "baselines" / "summary.json").read_text())
    return next(r[key] for r in rows if r["adv"] == adv and r["good"] == good and r["num_forests"] == 2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    p.add_argument("--labels", nargs="*")
    p.add_argument("--out", default="presentation/figures/training.pdf")
    args = p.parse_args()
    labels = args.labels or [Path(r).name for r in args.runs]
    color_of = {lab: COLORS[i] for i, lab in enumerate(dict.fromkeys(labels))}  # same label, same color

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), dpi=150)
    panels = [
        ("eval_adv_vs_heuristic_caught", "Drapieżnicy MAPPO vs heur. ofiary",
         ("heuristic", "heuristic"), "heurystyczni drapieżnicy"),
        ("eval_good_vs_heuristic_caught", "Ofiary MAPPO vs heur. drapieżnicy (mniej = lepiej)",
         ("heuristic", "heuristic"), "heurystyczne ofiary"),
    ]
    for ax, (col, title, (b_adv, b_good), b_label) in zip(axes, panels):
        for run, lab in zip(args.runs, labels):
            df = pd.read_csv(str(Path(run) / "progress.csv"))
            if col not in df:
                continue
            d = df.dropna(subset=[col])
            ax.plot(d.env_steps / 1e6, 100 * d[col], color=color_of[lab], lw=2, label=lab)
        ref = 100 * baseline(b_adv, b_good, "caught")
        ax.axhline(ref, color=MUTED, lw=1.2, ls="--", label=b_label)
        ax.set_ylim(0, 105)
        style_axes(ax, title, "% epizodów ze złapaniem")
    for ax, loc in zip(axes, ("lower right", "lower left")):
        ax.legend(fontsize=7.5, frameon=False, loc=loc)
    fig.tight_layout()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    main()
