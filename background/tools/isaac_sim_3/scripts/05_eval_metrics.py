"""Metrics-only evaluation loop, using a zero baseline (NOT a trained policy).

Run from IsaacLab checkout:
  uv run python /path/to/scripts/05_eval_metrics.py --episodes 100 --num_envs 32

Replace the function policy_actions() with inference from your trained checkpoint
using its original observation normalization/wrappers to evaluate a real policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config

parser = argparse.ArgumentParser(description="Evaluate episodes from a Gymnasium rollout")
parser.add_argument("--task", default="Isaac-Cartpole-Direct")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--output", default="outputs/zero_baseline.json")
add_launcher_args(parser)
args = parser.parse_args()


def policy_actions(obs: dict, env: gym.Env) -> torch.Tensor:
    """Baseline with zero actions, not a learned controller."""
    return torch.zeros(env.action_space.shape, device=env.unwrapped.device)


def main() -> None:
    cfg, _ = resolve_task_config(args.task, "", overrides=("physics=newton_mjwarp",))
    cfg.scene.num_envs = args.num_envs
    if args.device is not None:
        cfg.sim.device = args.device

    completed: list[dict] = []
    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg)
        try:
            obs, _ = env.reset()
            device = env.unwrapped.device
            running_returns = torch.zeros(args.num_envs, device=device)
            running_lengths = torch.zeros(args.num_envs, dtype=torch.int32, device=device)
            while len(completed) < args.episodes:
                with torch.inference_mode():
                    actions = policy_actions(obs, env)
                    obs, reward, terminated, truncated, info = env.step(actions)
                    running_returns += reward
                    running_lengths += 1
                    done = terminated | truncated
                    for idx in done.nonzero(as_tuple=False).flatten().tolist():
                        completed.append({
                            "return": float(running_returns[idx].item()),
                            "length": int(running_lengths[idx].item()),
                            "terminated": bool(terminated[idx].item()),
                            "truncated": bool(truncated[idx].item()),
                        })
                    running_returns[done] = 0.0
                    running_lengths[done] = 0
        finally:
            env.close()

    completed = completed[:args.episodes]
    summary = {
        "task": args.task,
        "controller": "zero_baseline_not_trained",
        "episodes": len(completed),
        "return_mean": sum(e["return"] for e in completed) / len(completed),
        "length_mean": sum(e["length"] for e in completed) / len(completed),
        "ended_by_timeout": sum(e["truncated"] for e in completed),
        "success_rate": None,  # No task-specific success signal defined here.
        "episode_results": completed,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "episode_results"}, indent=2))
    print("Saved:", path)


if __name__ == "__main__":
    main()
