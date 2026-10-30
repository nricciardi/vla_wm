"""Video recording through the Isaac Lab 3.0 configuration API, not Gym RecordVideo.

Run from IsaacLab root (RTX GPU, Isaac Sim 6.2):
  uv run --extra isaacsim --extra video python /path/to/scripts/04_video_gym.py --viz kit
"""
from __future__ import annotations

import argparse
import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description="Record MP4 from custom Gym loop")
parser.add_argument("--steps", type=int, default=125)
parser.add_argument("--output", default="videos/tutorial")
add_launcher_args(parser)
args, overrides = setup_preset_cli(parser)


def main() -> None:
    task = "Isaac-Cartpole-Direct"
    cfg, _ = resolve_task_config(task, "", overrides=(*overrides, "physics=isaacsim_physx"))
    cfg.scene.num_envs = 4
    if args.device is not None:
        cfg.sim.device = args.device
    cfg.video_recorders = [VideoRecorderCfg(
        source="visualizer:kit", output_dir=args.output,
        output_filename_prefix="cartpole_demo", video_length=100,
        video_interval=0, fps=30, step_offset=5,
    )]
    # Kit captures the viewport through its renderer.
    args.enable_cameras = True
    with launch_simulation(cfg, args):
        env = gym.make(task, cfg=cfg)
        try:
            obs, info = env.reset()
            for _ in range(args.steps):
                with torch.inference_mode():
                    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                    obs, reward, terminated, truncated, info = env.step(actions)
        finally:
            env.close()
    print("Video saved under:", args.output)


if __name__ == "__main__":
    main()
