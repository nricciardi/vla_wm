"""Build a table, add rigid or deformable objects, and simulate their motion.

Start reading at create_scene(): it creates the ground, the table, and each
object. Geometry, rigid-body physics, and deformable physics have separate
functions. JSON loading and validation live in scene_config.py.

Run from the Isaac Lab checkout (fragment API, Isaac Sim / PhysX, CUDA GPU):
    uv run --extra isaacsim --extra tetrahedralization python \
        /path/to/vla_wm/background/tools/isaac_sim_3/deformables/lab/06_deformable_tabletop.py --viz kit

Use --validate-only to check the scene without installing Isaac Lab.
API references:
    https://docs.nvidia.com/isaaclab/3.0/concepts/physics-schemas
    https://docs.nvidia.com/isaaclab/3.0/concepts/deformables
"""
from __future__ import annotations

import sys

from scene_config import ObjectSpec, TableSpec, load_scene, parse_arguments


def create_scene(table: TableSpec, object_specs: list[ObjectSpec]) -> dict:
    """Create the complete scene before the physics buffers are initialized."""
    create_ground_and_light()
    create_table(table)

    objects = {}

    for spec in object_specs:
        if spec.deformable:
            body = create_deformable_object(spec, table)
        else:
            body = create_rigid_object(spec, table)

        objects[spec.name] = body

    return objects


def create_ground_and_light() -> None:
    import isaaclab.sim as sim_utils

    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)

    light = sim_utils.DomeLightCfg(intensity=2500.0)
    light.func("/World/Light", light)


def create_table(table: TableSpec) -> None:
    """Create a static tabletop with its own collision and friction material."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.schemas import UsdPhysicsCollisionCfg
    from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg

    contact_material = UsdPhysicsRigidBodyMaterialCfg(
        static_friction=table.static_friction,
        dynamic_friction=table.dynamic_friction,
        restitution=0.0,
    )

    # A collider without a dynamic rigid body stays fixed under contact forces.
    tabletop = sim_utils.CuboidCfg(
        size=tuple(table.size),
        collision_props=UsdPhysicsCollisionCfg(collision_enabled=True),
        physics_material=contact_material,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(table.color)),
    )

    tabletop.func("/World/Table", tabletop, translation=tuple(table.center))


def create_rigid_object(spec: ObjectSpec, table: TableSpec):
    """Add geometry with rigid-body mass, collisions, and surface friction."""
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim.schemas import MassCfg, UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
    from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
    from isaaclab_physx.sim.schemas import PhysxConvexHullCfg

    contact_material = UsdPhysicsRigidBodyMaterialCfg(
        static_friction=spec.static_friction,
        dynamic_friction=spec.dynamic_friction,
        restitution=0.0,
    )

    physics = {
        "rigid_props": UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
        "mass_props": MassCfg(mass=spec.mass),
        "collision_props": UsdPhysicsCollisionCfg(collision_enabled=True),
        "physics_material": contact_material,
    }

    if spec.shape == "usd":
        physics["mesh_collision_props"] = PhysxConvexHullCfg()

    geometry = create_geometry(spec, **physics)

    config = RigidObjectCfg(
        prim_path=f"/World/Objects/{spec.name}",
        spawn=geometry,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=world_position(spec, table),
            rot=tuple(spec.rotation_xyzw),
        ),
    )

    return RigidObject(cfg=config)


def create_deformable_object(spec: ObjectSpec, table: TableSpec):
    """Add a volume deformable with explicit total mass and elastic material."""
    from isaaclab.assets import DeformableObject, DeformableObjectCfg
    from isaaclab.sim.schemas import OmniPhysicsDeformableBodyCfg
    from isaaclab.sim.spawners.materials import OmniPhysicsDeformableMaterialCfg
    from isaaclab_physx.sim.schemas import PhysxDeformableBodyCfg
    from isaaclab_physx.sim.spawners.materials import PhysxDeformableMaterialCfg

    # Deformable mass belongs to OmniPhysics, rather than rigid-body MassCfg.
    body_properties = [
        OmniPhysicsDeformableBodyCfg(mass=spec.mass),
        PhysxDeformableBodyCfg(solver_position_iteration_count=32),
    ]

    elastic_material = [
        OmniPhysicsDeformableMaterialCfg(
            youngs_modulus=spec.youngs_modulus,
            poissons_ratio=spec.poissons_ratio,
            static_friction=spec.static_friction,
            dynamic_friction=spec.dynamic_friction,
        ),
        PhysxDeformableMaterialCfg(elasticity_damping=spec.elasticity_damping),
    ]

    geometry = create_geometry(
        spec,
        volume_deformable_props=body_properties,
        physics_material=elastic_material,
    )

    config = DeformableObjectCfg(
        prim_path=f"/World/Objects/{spec.name}",
        spawn=geometry,
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=world_position(spec, table),
            rot=tuple(spec.rotation_xyzw),
        ),
    )

    return DeformableObject(cfg=config)


def create_geometry(spec: ObjectSpec, **physics):
    """Choose the shape spawner; imported USD assets keep their visual materials."""
    import isaaclab.sim as sim_utils

    if spec.shape == "usd":
        return sim_utils.UsdFileCfg(
            usd_path=spec.usd_path,
            scale=tuple(spec.scale),
            **physics,
        )

    physics["visual_material"] = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(spec.color))

    if spec.deformable:
        physics["edge_refinement"] = spec.edge_refinement

    if spec.shape == "box":
        spawner = sim_utils.MeshCuboidCfg if spec.deformable else sim_utils.CuboidCfg
        return spawner(size=tuple(spec.size), **physics)

    if spec.shape == "sphere":
        spawner = sim_utils.MeshSphereCfg if spec.deformable else sim_utils.SphereCfg
        return spawner(radius=spec.radius, **physics)

    if spec.shape == "cylinder":
        spawner = sim_utils.MeshCylinderCfg if spec.deformable else sim_utils.CylinderCfg
        return spawner(radius=spec.radius, height=spec.height, axis="Z", **physics)

    raise ValueError(f"Unsupported shape: {spec.shape}")


def world_position(spec: ObjectSpec, table: TableSpec) -> tuple[float, float, float]:
    """Convert a position above the tabletop surface to world coordinates."""
    table_x, table_y, table_z = table.center
    offset_x, offset_y, offset_z = spec.position
    surface_height = table_z + table.size[2] / 2

    return table_x + offset_x, table_y + offset_y, surface_height + offset_z


def set_initial_velocities(objects: dict, object_specs: list[ObjectSpec]) -> None:
    """Initialize motion once, allowing horizontal sliding experiments."""
    for spec in object_specs:
        body = objects[spec.name]

        if spec.deformable:
            state = body.data.default_nodal_state_w.torch.clone()
            state[..., 3:6] = state.new_tensor(spec.linear_velocity)

            body.write_nodal_state_to_sim_index(nodal_state=state)

        else:
            velocity = body.data.default_root_vel.torch.clone()
            velocity[:, :3] = velocity.new_tensor(spec.linear_velocity)

            body.write_root_velocity_to_sim_index(root_velocity=velocity)

        body.reset()


def simulate(sim, objects: dict, *, steps: int, dt: float, log_every: int) -> None:
    """Send commands, advance physics, and refresh the asset buffers."""
    import torch

    with torch.inference_mode():
        for step in range(steps):
            if not sim.is_running():
                break

            for body in objects.values():
                body.write_data_to_sim()

            sim.step()

            for body in objects.values():
                body.update(dt)

            if step % log_every == 0:
                heights = {
                    name: round(body.data.root_pos_w.torch[0, 2].item(), 3)
                    for name, body in objects.items()
                }

                print(f"step={step} simulated_time={(step + 1) * dt:.3f}s z_world={heights}")


def run_simulation(table: TableSpec, object_specs: list[ObjectSpec], args) -> None:
    """Launch PhysX, build the scene, initialize it, and run the experiment."""
    from isaaclab.app import launch_simulation
    from isaaclab.sim import SimulationCfg
    from isaaclab_physx.physics import PhysxCfg

    device = args.device or "cuda:0"

    if not device.startswith("cuda"):
        raise ValueError("This PhysX deformable example requires a CUDA device, such as --device cuda:0")

    config = SimulationCfg(dt=args.dt, device=device, physics=PhysxCfg())

    with launch_simulation(config, args):
        from isaaclab.sim import SimulationContext

        sim = SimulationContext(config)
        center_x, center_y, center_z = table.center

        sim.set_camera_view(
            eye=[center_x + 2.0, center_y + 1.8, center_z + 1.4],
            target=[center_x, center_y, center_z + 0.15],
        )

        # Build all scene geometry before initializing physics.
        objects = create_scene(table, object_specs)

        sim.reset()
        set_initial_velocities(objects, object_specs)

        simulate(sim, objects, steps=args.steps, dt=args.dt, log_every=args.log_every)


def main() -> None:
    args = parse_arguments(sys.argv[1:])

    try:
        table, object_specs = load_scene(args.config)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Invalid scene configuration: {exc}") from exc

    print(f"Valid scene: {len(object_specs)} objects; tabletop dimensions={tuple(table.size)} m")

    for spec in object_specs:
        print(
            f"  {spec.name}: shape={spec.shape}, deformable={spec.deformable}, "
            f"mass={spec.mass} kg, friction=({spec.static_friction}, {spec.dynamic_friction})"
        )

    if args.validate_only:
        return

    run_simulation(table, object_specs, args)


if __name__ == "__main__":
    main()
