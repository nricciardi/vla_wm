"""Collect episode metrics for a zero-action baseline, without a trained policy.

From the Isaac Lab checkout:
    uv run python /path/to/vla_wm/background/tools/isaac_sim_3/evaluation/lab/05_eval_metrics.py --episodes 100 --num_envs 32

Replace policy_actions() with checkpoint inference, keeping the observation
normalization and wrappers used during training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401 - Register the built-in Gymnasium tasks.
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config


def policy_actions(observations: dict, env: gym.Env) -> torch.Tensor:
    """Return a zero-action baseline; this function contains no learned policy."""
    return torch.zeros(env.action_space.shape, device=env.unwrapped.device)


def collect_episode_metrics(env: gym.Env, *, episodes: int, num_envs: int) -> list[dict]:
    """Keep separate return and length counters for each environment copy."""
    observations, info = env.reset()
    device = env.unwrapped.device

    running_returns = torch.zeros(num_envs, device=device)
    running_lengths = torch.zeros(num_envs, dtype=torch.int32, device=device)
    completed = []

    while len(completed) < episodes:
        with torch.inference_mode():
            actions = policy_actions(observations, env)

            observations, rewards, terminated, truncated, info = env.step(actions)

            running_returns += rewards
            running_lengths += 1
            finished = terminated | truncated

            # An environment can finish while the other copies keep running.
            finished_indices = finished.nonzero(as_tuple=False).flatten().tolist()

            for index in finished_indices:
                completed.append({
                    "return": float(running_returns[index].item()),
                    "length": int(running_lengths[index].item()),
                    "terminated": bool(terminated[index].item()),
                    "truncated": bool(truncated[index].item()),
                })

            running_returns[finished] = 0.0
            running_lengths[finished] = 0

    # Several copies may finish on the last step; retain the requested count.
    return completed[:episodes]


def save_report(completed: list[dict], *, task: str, output: Path) -> None:
    episode_count = len(completed)

    summary = {
        "task": task,
        "controller": "zero_baseline_not_trained",
        "episodes": episode_count,
        "return_mean": sum(episode["return"] for episode in completed) / episode_count,
        "length_mean": sum(episode["length"] for episode in completed) / episode_count,
        "ended_by_timeout": sum(episode["truncated"] for episode in completed),
        "success_rate": None,  # Success requires a task-specific condition.
        "episode_results": completed,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    printed_summary = {key: value for key, value in summary.items() if key != "episode_results"}

    print(json.dumps(printed_summary, indent=2))
    print("Saved:", output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Gymnasium episode metrics")
    parser.add_argument("--task", default="Isaac-Cartpole-Direct")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("outputs/zero_baseline.json"))

    add_launcher_args(parser)
    args = parser.parse_args()

    if args.episodes < 1 or args.num_envs < 1:
        parser.error("episodes and num_envs must be positive")

    config, _ = resolve_task_config(
        args.task,
        "",
        overrides=("physics=newton_mjwarp",),
    )
    config.scene.num_envs = args.num_envs

    if args.device is not None:
        config.sim.device = args.device

    with launch_simulation(config, args):
        env = gym.make(args.task, cfg=config)

        try:
            completed = collect_episode_metrics(env, episodes=args.episodes, num_envs=args.num_envs)
        finally:
            env.close()

    save_report(completed, task=args.task, output=args.output)


if __name__ == "__main__":
    main()
