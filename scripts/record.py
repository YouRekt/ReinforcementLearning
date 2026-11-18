"""Record episodes of simple_world_comm to mp4 (plus a frame strip for slides).

Policies per team: random | heuristic | mappo:<checkpoint.pt>
  uv run python scripts/record.py --adv random --good random --out presentation/media/random_agents.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from worldcomm import EpisodeStats, make_env, role_of, LEADER
from worldcomm.env import MOVE_NAMES, split_leader_action
from worldcomm.policies import TeamMix, make_policy
from worldcomm.render import caught_agents, load_font, render_world

ROOT = Path(__file__).resolve().parents[1]
WORLD_PX = 600
CANVAS = (928, 608)  # multiples of 16 for ffmpeg
FONT, SMALL = load_font(19), load_font(16)

TEAM_LABEL = {"random": "losowa", "heuristic": "heurystyczna"}


def label(spec: str) -> str:
    return TEAM_LABEL.get(spec, "MAPPO" if spec.startswith("mappo") else spec)


def frame(env, t, actions, stats, adv_spec, good_spec, done=False) -> np.ndarray:
    world = env.unwrapped.world
    canvas = Image.new("RGB", CANVAS, "white")
    canvas.paste(Image.fromarray(render_world(world, WORLD_PX, caught_agents(world))), (0, 4))
    d = ImageDraw.Draw(canvas)
    x, y = WORLD_PX + 16, 16
    lines = [
        (f"Krok: {t} / {env.unwrapped.max_cycles}", FONT, "black"),
        ("", SMALL, "black"),
        (f"Drapieżnicy: {label(adv_spec)}", FONT, (170, 40, 40)),
        (f"  suma nagród: {stats.team_return(True):.1f}", SMALL, "black"),
        (f"Ofiary: {label(good_spec)}", FONT, (40, 130, 40)),
        (f"  suma nagród: {stats.team_return(False):.1f}", SMALL, "black"),
        ("", SMALL, "black"),
        (f"Złapania: {stats.catches}", FONT, "black"),
    ]
    if actions:
        lead = next(a for a in actions if role_of(a) == LEADER)
        move, say = split_leader_action(int(actions[lead]))
        lines.append((f"Lider: ruch {MOVE_NAMES[move]}, komunikat {say}", SMALL, "black"))
    for text, font, color in lines:
        d.text((x, y), text, font=font, fill=color)
        y += 30 if font is FONT else 24
    y = CANVAS[1] - 120
    for color, text in [((140, 20, 20), "L = lider drapieżników"), ((225, 90, 90), "drapieżnik"),
                        ((60, 170, 60), "ofiara"), ((40, 60, 190), "jedzenie"), ((120, 190, 120), "las")]:
        d.ellipse([x, y + 3, x + 14, y + 17], fill=color)
        d.text((x + 22, y), text, font=SMALL, fill="black")
        y += 22
    if done:
        d.text((x, CANVAS[1] - 150), "KONIEC EPIZODU", font=FONT, fill=(170, 40, 40))
    return np.asarray(canvas)


def run(env, policy, seed, adv_spec, good_spec):
    obs, _ = env.reset(seed=seed)
    st = EpisodeStats()
    st.reset(env)
    frames = [frame(env, 0, None, st, adv_spec, good_spec)]
    while env.agents:
        actions = policy(obs)
        obs, rew, _, _, _ = env.step(actions)
        st.step(env, actions, rew)
        frames.append(frame(env, st.steps, actions, st, adv_spec, good_spec, done=not env.agents))
    print(f"seed {seed}: {st.finish()}")
    return frames


def save_strip(frames, path, n=5):
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    tiles = [Image.fromarray(frames[i][4:4 + WORLD_PX, :WORLD_PX]).resize((300, 300), Image.Resampling.LANCZOS) for i in idx]
    strip = Image.new("RGB", (n * 300 + (n - 1) * 8, 332), "white")
    d = ImageDraw.Draw(strip)
    for k, (tile, i) in enumerate(zip(tiles, idx)):
        strip.paste(tile, (k * 308, 32))
        d.text((k * 308 + 4, 4), f"t = {i}", font=FONT, fill="black")
    strip.save(path)
    print("saved", path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adv", default="random")
    p.add_argument("--good", default="random")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--out", default="presentation/media/random_agents.mp4")
    p.add_argument("--strip", default="presentation/figures/random_strip.png")
    args = p.parse_args()

    env = make_env()
    policy = TeamMix(make_policy(args.adv, env), make_policy(args.good, env))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8)
    try:
        for k, seed in enumerate(args.seeds):
            frames = run(env, policy, seed, args.adv, args.good)
            if k == 0 and args.strip:
                save_strip(frames, ROOT / args.strip)
            for f in frames + [frames[-1]] * args.fps:
                w.append_data(f)
    finally:
        w.close()
    print("saved", out)


if __name__ == "__main__":
    main()
