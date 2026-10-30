"""Customize Cartpole rewards while reusing the built-in DirectRLEnv scene.

From the Isaac Lab checkout:
    uv run python /path/to/vla_wm/background/tools/isaac_sim_3/custom_tasks/lab/03_custom_cartpole.py --viz newton_gl

This example uses zero actions. It defines a new task but does not train a policy.
"""
from __future__ import annotations

import argparse

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401 - Register the built-in task configurations.
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import configclass
from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv
from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg
from isaaclab_tasks.utils import resolve_task_config

GYM_ID = "Tutorial-Cartpole-Custom-v0"


@configclass
class MyCartpoleCfg(CartpoleEnvCfg):
    """Keep the original scene and change the episode duration and reward weights."""

    episode_length_s = 8.0

    upright_bonus = 2.0
    cart_penalty = 0.10


class MyCartpoleEnv(CartpoleEnv):
    cfg: MyCartpoleCfg

    def _get_rewards(self) -> torch.Tensor:
        pole_angle = self.cartpole.data.joint_pos.torch[:, self._pole_dof_idx].squeeze(-1)
        cart_position = self.cartpole.data.joint_pos.torch[:, self._cart_dof_idx].squeeze(-1)

        # Reward an upright pole and penalize cart displacement and task failure.
        upright_reward = self.cfg.upright_bonus * torch.cos(pole_angle)
        displacement_penalty = self.cfg.cart_penalty * cart_position.square()
        termination_penalty = 2.0 * self.reset_terminated.float()

        return upright_reward - displacement_penalty - termination_penalty


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a custom Cartpole reward example")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)

    add_launcher_args(parser)
    args = parser.parse_args()

    # Reuse the backend configuration of the original task.
    base_config, _ = resolve_task_config(
        "Isaac-Cartpole-Direct",
        "",
        overrides=("physics=newton_mjwarp",),
    )

    config = MyCartpoleCfg()
    config.sim = base_config.sim
    config.scene.num_envs = args.num_envs

    if args.device is not None:
        config.sim.device = args.device

    # Registration is local to this Python process.
    gym.register(id=GYM_ID, entry_point=MyCartpoleEnv, disable_env_checker=True)

    with launch_simulation(config, args):
        env = gym.make(GYM_ID, cfg=config)

        try:
            observations, info = env.reset()

            for step in range(args.steps):
                with torch.inference_mode():
                    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

                    observations, rewards, terminated, truncated, info = env.step(actions)

                if step % 50 == 0:
                    print(f"step={step} mean_reward={rewards.mean().item():.4f}")

        finally:
            env.close()


if __name__ == "__main__":
    main()
