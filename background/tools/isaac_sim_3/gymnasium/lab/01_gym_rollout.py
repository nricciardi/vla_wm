"""Run a Gymnasium rollout on an Isaac Lab 3.0 Cartpole task using Newton.

From the Isaac Lab checkout:
    uv run python /path/to/vla_wm/background/tools/isaac_sim_3/gymnasium/lab/01_gym_rollout.py --viz newton_gl
"""
from __future__ import annotations

import argparse

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401 - Register the built-in Gymnasium tasks.
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batched Gymnasium rollout")
    parser.add_argument("--task", default="Isaac-Cartpole-Direct")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)

    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser)

    # Resolve the task description before creating the simulation runtime.
    config, _ = resolve_task_config(
        args.task,
        "",
        overrides=(*overrides, "physics=newton_mjwarp"),
    )
    config.scene.num_envs = args.num_envs

    if args.device is not None:
        config.sim.device = args.device

    with launch_simulation(config, args):
        env = gym.make(args.task, cfg=config)

        try:
            observations, info = env.reset()

            print("action_space:", env.action_space)
            print("observation_space:", env.observation_space)
            print("policy observations:", observations["policy"].shape)

            for step in range(args.steps):
                with torch.inference_mode():
                    # One command per environment, on the simulation device.
                    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                    actions[:, 0] = 0.25 if (step // 40) % 2 == 0 else -0.25

                    observations, rewards, terminated, truncated, info = env.step(actions)

                if step % 50 == 0:
                    finished_count = (terminated | truncated).sum().item()

                    print(
                        f"step={step:04d} reward_mean={rewards.mean().item():+.3f} "
                        f"done={finished_count}/{args.num_envs}"
                    )

        finally:
            env.close()


if __name__ == "__main__":
    main()
