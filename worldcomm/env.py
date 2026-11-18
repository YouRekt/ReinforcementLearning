"""Environment factory and layout constants for MPE2 simple_world_comm_v3."""

from __future__ import annotations

from mpe2 import simple_world_comm_v3

ENV_NAME = "simple_world_comm_v3"

MOVE_NAMES = ["stop", "left", "right", "down", "up"]
N_MOVES = 5
N_SAY = 4  # leader's communication channel size (world.dim_c)

# Roles: one leader, normal adversaries, good agents (prey)
LEADER, ADVERSARY, GOOD = "leader", "adversary", "good"


def make_env(
    num_good: int = 2,
    num_adversaries: int = 4,
    num_obstacles: int = 1,
    num_food: int = 2,
    num_forests: int = 2,
    max_cycles: int = 25,
    render_mode: str | None = None,
):
    """Parallel (all agents act at once) PettingZoo env with discrete actions."""
    return simple_world_comm_v3.parallel_env(
        num_good=num_good,
        num_adversaries=num_adversaries,
        num_obstacles=num_obstacles,
        num_food=num_food,
        num_forests=num_forests,
        max_cycles=max_cycles,
        continuous_actions=False,
        render_mode=render_mode,
    )


def role_of(agent: str) -> str:
    if agent.startswith("leadadversary"):
        return LEADER
    if agent.startswith("adversary"):
        return ADVERSARY
    return GOOD


def is_adversary_team(agent: str) -> bool:
    return role_of(agent) != GOOD


def split_leader_action(action: int) -> tuple[int, int]:
    """Leader's Discrete(20) action -> (move, say); the env decodes move = a % 5."""
    return action % N_MOVES, action // N_MOVES
