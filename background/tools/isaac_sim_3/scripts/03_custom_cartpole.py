"""A genuine custom DirectRLEnv subclass registered with Gymnasium.

We reuse NVIDIA's shipped Cartpole scene so that the exercise focuses on the
custom environment contract (reward/done/reset) rather than robot asset import.

Run: uv run python /path/to/scripts/03_custom_cartpole.py --viz newton_gl
"""
from __future__ import annotations

import argparse
import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import configclass
from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv
from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg
from isaaclab_tasks.utils import resolve_task_config


@configclass
class MyCartpoleCfg(CartpoleEnvCfg):
    """Same robot/scene, but different episode duration and task reward."""
    episode_length_s = 8.0
    upright_bonus = 2.0
    cart_penalty = 0.10


class MyCartpoleEnv(CartpoleEnv):
    cfg: MyCartpoleCfg

    def _get_rewards(self) -> torch.Tensor:
        # Positive reward for a vertical pole, penalty for moving the cart.
        angle = self.cartpole.data.joint_pos.torch[:, self._pole_dof_idx].squeeze(-1)
        cart = self.cartpole.data.joint_pos.torch[:, self._cart_dof_idx].squeeze(-1)
        reward = self.cfg.upright_bonus * torch.cos(angle) - self.cfg.cart_penalty * cart.square()
        reward -= 2.0 * self.reset_terminated.float()
        return reward


GYM_ID = "Tutorial-Cartpole-Custom-v0"
gym.register(id=GYM_ID, entry_point=MyCartpoleEnv, disable_env_checker=True)

parser = argparse.ArgumentParser(description="Custom Gymnasium task")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=200)
add_launcher_args(parser)
args = parser.parse_args()


def main() -> None:
    # Resolve a known-good backend preset from the original task, then apply
    # our custom task configuration. The official config object has a physics
    # backend selected by the Newton preset.
    base_cfg, _ = resolve_task_config("Isaac-Cartpole-Direct", "", overrides=("physics=newton_mjwarp",))
    cfg = MyCartpoleCfg()
    cfg.sim = base_cfg.sim
    cfg.scene.num_envs = args.num_envs
    if args.device is not None:
        cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        env = gym.make(GYM_ID, cfg=cfg)
        try:
            obs, _ = env.reset()
            for t in range(args.steps):
                with torch.inference_mode():
                    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                    obs, rew, terminated, truncated, info = env.step(actions)
                if t % 50 == 0:
                    print(f"step={t} mean_reward={rew.mean().item():.4f}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
