"""Create a static tabletop and three rigid cubes with Isaac Sim / PhysX.

From the Isaac Lab checkout:
    uv run --extra isaacsim python \
        /path/to/vla_wm/background/tools/isaac_sim_3/scenes_and_objects/lab/02_tabletop_objects.py --viz kit

All geometry is procedural. Cube mass and friction are command-line arguments.
"""
from __future__ import annotations

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim.schemas import MassCfg, UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg


def create_scene(*, mass: float, friction: float) -> list[RigidObject]:
    """Create the fixed environment, then place three cubes above the table."""
    create_ground_and_light()
    create_table()

    cubes = []

    for index, x_position in enumerate((-0.32, 0.0, 0.32)):
        cube = create_cube(
            name=f"Cube{index}",
            position=(x_position, 0.0, 1.15),
            mass=mass * (index + 1),
            friction=friction,
            color=(0.85 if index == 0 else 0.15, 0.2 + index * 0.3, 0.25),
        )

        cubes.append(cube)

    return cubes


def create_ground_and_light() -> None:
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)

    light = sim_utils.DomeLightCfg(intensity=2500.0)
    light.func("/World/Light", light)


def create_table() -> None:
    """Create a static collider whose top surface is at z=0.80 m."""
    tabletop = sim_utils.CuboidCfg(
        size=(1.2, 0.8, 0.10),
        collision_props=UsdPhysicsCollisionCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.50, 0.35, 0.22)),
    )

    tabletop.func("/World/Table", tabletop, translation=(0.0, 0.0, 0.75))


def create_cube(
    *,
    name: str,
    position: tuple[float, float, float],
    mass: float,
    friction: float,
    color: tuple[float, float, float],
) -> RigidObject:
    """Create one dynamic cube; lengths are in meters and mass is in kg."""
    contact_material = RigidBodyMaterialBaseCfg(
        static_friction=friction,
        dynamic_friction=min(friction, 0.5),
        restitution=0.05,
    )

    geometry = sim_utils.CuboidCfg(
        size=(0.10, 0.10, 0.10),
        rigid_props=UsdPhysicsRigidBodyCfg(),
        mass_props=MassCfg(mass=mass),
        collision_props=UsdPhysicsCollisionCfg(collision_enabled=True),
        physics_material=contact_material,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )

    config = RigidObjectCfg(
        prim_path=f"/World/{name}",
        spawn=geometry,
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )

    return RigidObject(cfg=config)


def simulate(sim, cubes: list[RigidObject], *, steps: int) -> None:
    dt = sim.get_physics_dt()

    for step in range(steps):
        if not sim.is_running():
            break

        for cube in cubes:
            cube.write_data_to_sim()

        sim.step()

        for cube in cubes:
            cube.update(dt)

        if step % 60 == 0:
            heights = [round(cube.data.root_pos_w.torch[0, 2].item(), 3) for cube in cubes]

            print(f"step={step} z_positions={heights}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tabletop and configurable rigid cubes")
    parser.add_argument("--mass", type=float, default=0.5, help="Base cube mass [kg]")
    parser.add_argument("--friction", type=float, default=0.65, help="Static friction coefficient")
    parser.add_argument("--steps", type=int, default=600)

    add_launcher_args(parser)
    args = parser.parse_args()

    config = sim_utils.SimulationCfg(dt=1.0 / 120, device=args.device)

    with launch_simulation(config, args):
        sim = sim_utils.SimulationContext(config)
        sim.set_camera_view(eye=[2.0, 1.7, 1.8], target=[0.0, 0.0, 0.75])

        # Create all physical bodies before initializing the physics buffers.
        cubes = create_scene(mass=args.mass, friction=args.friction)

        sim.reset()
        simulate(sim, cubes, steps=args.steps)


if __name__ == "__main__":
    main()
