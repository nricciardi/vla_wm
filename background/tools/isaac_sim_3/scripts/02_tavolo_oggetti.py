"""Isaac Sim 6.2 / PhysX: procedural table, cubes with adjustable mass and friction.

Run from the IsaacLab root:
  uv run --extra isaacsim python /path/to/scripts/02_tavolo_oggetti.py --viz kit
No external USD models are needed; the tabletop is a static cuboid collider.
"""
from __future__ import annotations

import argparse
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Tavolo e cubi modificabili")
parser.add_argument("--mass", type=float, default=0.5, help="Cube mass [kg]")
parser.add_argument("--friction", type=float, default=0.65, help="Static friction")
parser.add_argument("--steps", type=int, default=600)
add_launcher_args(parser)
args = parser.parse_args()

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim.schemas import MassCfg, UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg


def spawn_scene() -> list[RigidObject]:
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0)
    light.func("/World/Light", light)

    # Static table top: collider but no RigidBody physics. Center z=0.75; top z=0.80.
    table = sim_utils.CuboidCfg(
        size=(1.2, 0.8, 0.10),
        collision_props=UsdPhysicsCollisionCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.50, 0.35, 0.22)),
    )
    table.func("/World/Table", table, translation=(0.0, 0.0, 0.75))

    bodies = []
    for i, x in enumerate((-0.32, 0.0, 0.32)):
        cube_cfg = RigidObjectCfg(
            prim_path=f"/World/Cube{i}",
            spawn=sim_utils.CuboidCfg(
                size=(0.10, 0.10, 0.10),
                rigid_props=UsdPhysicsRigidBodyCfg(),
                mass_props=MassCfg(mass=args.mass * (i + 1)),
                collision_props=UsdPhysicsCollisionCfg(collision_enabled=True),
                physics_material=RigidBodyMaterialBaseCfg(
                    static_friction=args.friction,
                    dynamic_friction=min(args.friction, 0.5),
                    restitution=0.05,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=((0.85 if i == 0 else 0.15), 0.2 + i * 0.3, 0.25)
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, 0.0, 1.15)),
        )
        bodies.append(RigidObject(cfg=cube_cfg))
    return bodies


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120, device=args.device)
    with launch_simulation(sim_cfg, args):
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=[2.0, 1.7, 1.8], target=[0.0, 0.0, 0.75])
        bodies = spawn_scene()  # Create all physics bodies BEFORE sim.reset().
        sim.reset()
        dt = sim.get_physics_dt()
        for step in range(args.steps):
            if not sim.is_running():
                break
            for body in bodies:
                body.write_data_to_sim()
            sim.step()
            for body in bodies:
                body.update(dt)
            if step % 60 == 0:
                print("step", step, "z positions:",
                      [round(b.data.root_pos_w.torch[0, 2].item(), 3) for b in bodies])


if __name__ == "__main__":
    main()
