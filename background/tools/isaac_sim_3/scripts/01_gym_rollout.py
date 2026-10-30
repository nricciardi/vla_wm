"""Gymnasium custom loop on an Isaac Lab 3.0 task (Newton backend).

Run from the IsaacLab 3.0 root:
  uv run python /path/to/vla_wm/background/tools/isaac_sim_3/scripts/01_gym_rollout.py --viz newton_gl
"""
from __future__ import annotations

import argparse
import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401 - registers Gymnasium tasks
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description="Gymnasium loop, Isaac Lab 3.0")
parser.add_argument("--task", default="Isaac-Cartpole-Direct")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=300)
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)


def main() -> None:
    # The task owns its default backend; explicitly select Newton here.
    cfg, _ = resolve_task_config(args.task, "", overrides=(*hydra_args, "physics=newton_mjwarp"))
    cfg.scene.num_envs = args.num_envs
    if args.device is not None:
        cfg.sim.device = args.device

    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg)
        try:
            obs, info = env.reset()
            print("action_space:", env.action_space)
            print("observation_space:", env.observation_space)
            print("policy observations:", obs["policy"].shape)
            for t in range(args.steps):
                with torch.inference_mode():
                    # PyTorch tensor on the simulation device; one action per environment.
                    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                    actions[:, 0] = 0.25 if (t // 40) % 2 == 0 else -0.25
                    obs, rewards, terminated, truncated, info = env.step(actions)
                if t % 50 == 0:
                    print(f"step={t:04d} reward_mean={rewards.mean().item():+.3f} "
                          f"done={(terminated | truncated).sum().item()}/{args.num_envs}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
