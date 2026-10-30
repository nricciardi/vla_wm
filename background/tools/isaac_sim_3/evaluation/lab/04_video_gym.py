"""Record a zero-action Cartpole rollout using the Isaac Lab 3.0 video recorder.

From the Isaac Lab checkout (Isaac Sim / RTX):
    uv run --extra isaacsim --extra video python \
        /path/to/vla_wm/background/tools/isaac_sim_3/evaluation/lab/04_video_gym.py --viz kit
"""
from __future__ import annotations

import argparse

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401 - Register the built-in Gymnasium tasks.
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli


def main() -> None:
    parser = argparse.ArgumentParser(description="Record an MP4 from a Gymnasium rollout")
    parser.add_argument("--steps", type=int, default=125)
    parser.add_argument("--output", default="videos/tutorial")

    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser)

    task = "Isaac-Cartpole-Direct"
    config, _ = resolve_task_config(
        task,
        "",
        overrides=(*overrides, "physics=isaacsim_physx"),
    )
    config.scene.num_envs = 4

    if args.device is not None:
        config.sim.device = args.device

    # Configure the recorder before constructing the environment.
    recorder = VideoRecorderCfg(
        source="visualizer:kit",
        output_dir=args.output,
        output_filename_prefix="cartpole_demo",
        video_length=100,
        video_interval=0,
        fps=30,
        step_offset=5,
    )
    config.video_recorders = [recorder]
    args.enable_cameras = True

    with launch_simulation(config, args):
        env = gym.make(task, cfg=config)

        try:
            observations, info = env.reset()

            for _ in range(args.steps):
                with torch.inference_mode():
                    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

                    observations, rewards, terminated, truncated, info = env.step(actions)

        finally:
            env.close()

    print("Video saved under:", args.output)


if __name__ == "__main__":
    main()
