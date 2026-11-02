"""Fixed-camera renderer for simple_world_comm (the built-in one rescales every frame
and draws forests on top of agents, which hides them)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIEW = 1.6  # world coordinates shown: [-VIEW, VIEW]^2 (arena is [-1, 1]^2)
BG = (250, 250, 248)
COLORS = {
    "forest": (120, 190, 120, 110),
    "obstacle": (70, 70, 70, 255),
    "food": (40, 60, 190, 255),
    "leader": (140, 20, 20, 255),
    "adversary": (225, 90, 90, 255),
    "good": (60, 170, 60, 255),
}


@lru_cache(maxsize=None)
def load_font(size: int):
    for name in ("Arial Unicode.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def render_world(world, size: int = 600, caught: set[str] | None = None, view: float = VIEW) -> np.ndarray:
    """Draw the world state in [-view, view]^2; agents in `caught` get a black ring."""
    caught = caught or set()
    scale = size / (2 * view)

    def px(p):
        return (p[0] + view) * scale, (view - p[1]) * scale

    def circle(draw, pos, r, color, outline=None, width=1):
        cx, cy = px(pos)
        rr = r * scale
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=color, outline=outline, width=width)

    img = Image.new("RGBA", (size, size), BG + (255,))
    d = ImageDraw.Draw(img)
    a0, a1 = px((-1, 1)), px((1, -1))
    d.rectangle([a0, a1], outline=(160, 160, 160), width=2)  # arena edge (penalty outside)

    for lm in world.landmarks:
        if lm in world.forests or lm in world.food:
            continue
        circle(d, lm.state.p_pos, lm.size, COLORS["obstacle"])
    for f in world.food:
        circle(d, f.state.p_pos, f.size * 1.6, COLORS["food"])
    for ag in world.agents:
        role = "leader" if getattr(ag, "leader", False) else ("adversary" if ag.adversary else "good")
        ring = (0, 0, 0, 255) if ag.name in caught else None
        circle(d, ag.state.p_pos, ag.size, COLORS[role], outline=ring, width=3)
        if role == "leader":
            x, y = px(ag.state.p_pos)
            d.text((x, y), "L", fill="white", anchor="mm", font=load_font(int(ag.size * scale)))

    # forests last, translucent, so agents underneath stay visible
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(over)
    for f in world.forests:
        circle(od, f.state.p_pos, f.size, COLORS["forest"], outline=(60, 130, 60, 200), width=2)
    img = Image.alpha_composite(img, over)
    return np.asarray(img.convert("RGB"))


def caught_agents(world) -> set[str]:
    advs = [a for a in world.agents if a.adversary]
    out = set()
    for g in (a for a in world.agents if not a.adversary):
        for adv in advs:
            if np.linalg.norm(g.state.p_pos - adv.state.p_pos) < g.size + adv.size:
                out.update({g.name, adv.name})
    return out
