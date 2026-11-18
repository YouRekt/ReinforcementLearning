"""Shared matplotlib style for the learning-curve figures."""

from __future__ import annotations

INK, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#d43d8f"]


def style_axes(ax, title: str, ylabel: str, xlabel: str = "kroki środowiska (mln)"):
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7.5, colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
